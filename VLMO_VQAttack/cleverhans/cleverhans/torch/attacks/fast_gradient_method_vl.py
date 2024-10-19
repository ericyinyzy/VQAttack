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
  # rescaled=F.resize(input_tensor,(1,3,newh, neww))
  h_rem = image_height - newh
  w_rem = image_width - neww
  pad_top = int(np.random.uniform( 0, h_rem))
  pad_bottom = h_rem - pad_top
  pad_left = int(np.random.uniform( 0, w_rem))
  pad_right = w_rem - pad_left
  padding = (
      pad_left, pad_right,  # 前面填充1个单位，后面填充两个单位，输入的最后一个维度则增加1+2个单位，成为8
      pad_top, pad_bottom,
  )
  padded = F.pad(rescaled, padding)
  if padded.shape[-1]!= input_tensor.shape[-1] or padded.shape[-2]!= input_tensor.shape[-2]:
      raise ValueError
  # print('padded',padded[0,0,:32,:32],padded.shape)
  # exit()
  return padded
# np.random.seed(0)
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
    text_emb_pick=None
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
            torch.ge(x[0], torch.tensor(clip_min, device=x[0].device, dtype=x[0].dtype))
        )
        asserts.append(assert_ge)

    if clip_max is not None:
        assert_le = torch.all(
            torch.le(x[0], torch.tensor(clip_max, device=x[0].device, dtype=x[0].dtype))
        )
        asserts.append(assert_le)

    # x needs to be a leaf variable, of floating point type and have requires_grad being True for
    # its grad to be computed and stored properly in a backward call
    x[0] = x[0].clone().detach().to(torch.float).requires_grad_(True)
    x[1] = x[1].clone().detach().to(torch.float).requires_grad_(True)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos1 = nn.CosineSimilarity(dim=2, eps=1e-6)
    out = model_fn([x[0],x[1]])
    ###############
    #########################feature loss
    feat_len=min(out[2].shape[1],y[2].shape[1])
    out[2]=out[2][:,:feat_len,:]
    y[2]=y[2][:,:feat_len,:]
    if ls==1:
        loss=(torch.sum((-cos(out[1], y[1]) + torch.sum(-cos1(out[2], y[2]), 1))))
    elif ls==0:
        loss = (
            F.cross_entropy(out[0].view(-1, 30522),y[0].view(-1),ignore_index = -100)
        )
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    # torch.use_deterministic_algorithms(True, warn_only=True)
    torch.use_deterministic_algorithms(False)
    loss.requires_grad_(True).backward()
    optimal_perturbation = optimize_linear(x[0].grad,eps , norm)

    # Add perturbation to original example to obtain adversarial example
    adv_x = x[0] + optimal_perturbation
    text_gradient=x[1].grad[:,text_emb_pick]
    if (clip_min is not None) or (clip_max is not None):
        if clip_min is None or clip_max is None:
            raise ValueError(
                "One of clip_min and clip_max is None but we don't currently support one-sided clipping"
            )
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if sanity_checks:
        assert np.all(asserts)
    # print(torch.max(adv_x-x),torch.min(adv_x-x))
    return adv_x,text_gradient
