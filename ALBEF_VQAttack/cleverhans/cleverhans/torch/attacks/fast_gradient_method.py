"""The Fast Gradient Method attack."""
import numpy as np
import torch
import torch.nn as nn

from cleverhans.torch.utils import optimize_linear
import torch.nn.functional as F
from torchvision import transforms
def input_diversity(input_tensor):
  # input_tensor=copy.deepcopy(input)
  image_height=input_tensor.shape[-2]
  image_width=input_tensor.shape[-1]
  newh=int(np.random.uniform(image_height-32,image_height))
  neww=int((newh/image_height)*image_width)
  rescaled = transforms.functional.resize(input_tensor, (newh,neww))
  h_rem = image_height - newh
  w_rem = image_width - neww
  pad_top = int(np.random.uniform( 0, h_rem))
  pad_bottom = h_rem - pad_top
  pad_left = int(np.random.uniform( 0, w_rem))
  pad_right = w_rem - pad_left
  padding = (
      pad_left, pad_right,
      pad_top, pad_bottom,
  )
  padded = F.pad(rescaled, padding)
  if padded.shape[-1]!= input_tensor.shape[-1] or padded.shape[-2]!= input_tensor.shape[-2]:
      raise ValueError
  return padded
def fast_gradient_method(
    model_fn,
    x,
    eps,
    norm,
    ori_x,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    sanity_checks=False,
    ls=None,
        bkp=None,
        bkp_y=None
):
    """
    PyTorch implementation of the Fast Gradient Method.
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError(
            "Norm order must be either np.inf, 1, or 2, got {} instead.".format(norm)
        )
    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    if eps == 0:
        return x
    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        assert_ge = torch.all(
            torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_ge)

    if clip_max is not None:
        assert_le = torch.all(
            torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_le)
    x = x.clone().detach().to(torch.float).requires_grad_(True)
    cos1 = nn.CosineSimilarity(dim=2, eps=1e-6)
    out = model_fn(x)

    ###############################aaai loss#########################
    if ls==0:
        outshape = out[0].view(-1, 30522).shape[0]
        # print(outshape)
        if len(y[0].size()) == 2:
            if y[0].view(-1).shape[0]!= outshape:
                print('false to mlmq')
                ls=1
                out=bkp(x)
                y=bkp_y
        if len(y[0].size()) == 3:
            for i in range(y[0].shape[1]):
                if y[0][:, i, :].contiguous().view(-1).shape[0]!= outshape:
                    ls=1
                    print('false to mlmw')
                    out = bkp(x)
                    y = bkp_y
                    break
    #
    if ls==1:
        feat_len = min(out[1].shape[1], y[1].shape[1])
        out[1] = out[1][:, :feat_len, :]
        y[1] = y[1][:, :feat_len, :]
        feat_len_0 = min(out[0].shape[1], y[0].shape[1])
        out[0] = out[0][:, :feat_len_0, :]
        y[0] = y[0][:, :feat_len_0, :]
        loss=(torch.sum(torch.sum(-cos1(out[1], y[1]), 1)+torch.sum(-cos1(out[0], y[0]), 1)))
    elif ls==0:


        if len(y[0].size())==2:
            loss = (
                F.cross_entropy(out[0].view(-1, 30522),y[0].view(-1),ignore_index = -100)

            )
        elif len(y[0].size())==3:
            loss=F.cross_entropy(out[0].view(-1, 30522), y[0][:,0,:].view(-1), ignore_index=-100)
            for i in range(y[0].size()[1]-1):
                loss+=F.cross_entropy(out[0].view(-1, 30522), y[0][:,i+1,:].view(-1), ignore_index=-100)

        else:
            raise ValueError


    if targeted:
        loss = -loss

    # Define gradient of loss wrt input

    loss.requires_grad_(True).backward()
    optimal_perturbation = optimize_linear(x.grad,eps , norm)
    adv_x = x + optimal_perturbation

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
        if clip_min is None or clip_max is None:
            raise ValueError(
                "One of clip_min and clip_max is None but we don't currently support one-sided clipping"
            )
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if sanity_checks:
        assert np.all(asserts)
    # print(torch.max(adv_x-x),torch.min(adv_x-x))
    return adv_x,loss
