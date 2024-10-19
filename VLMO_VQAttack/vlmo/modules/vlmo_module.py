import os
# python run.py with task_finetune_vqa_base_image480 test_only=True
import copy
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import GPUtil as GPU
import json
import tensorflow_hub as hub
# from torch.nn import functional as F
from transformers import BertForMaskedLM, BertTokenizer
import vlmo.modules.multiway_transformer

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vlmo.modules import heads, objectives, vlmo_utils
from pytorch_lightning.utilities.distributed import rank_zero_info
from scipy import interpolate
from timm.models import create_model
from vlmo.modules.filter_words import filter_words
import nltk
from nltk.corpus import stopwords
import sys

sys.path.append('cleverhans')
import cleverhans.torch.attacks.projected_gradient_descent as pgd
import cleverhans.torch.attacks.projected_gradient_descent_vl as pgd_vl

nltk.download('stopwords')
filter_words = filter_words + stopwords.words('english')


class Feature(object):
    def __init__(self, seq_a):
        # self.label = label
        self.seq = seq_a
        self.final_adverse = seq_a
        self.query = 0
        self.change = 0
        self.success = 0
        self.sim = 0.0
        self.changes = []


def convert_to_textpt_ckpt(state_dict, module):
    new_state_dict = {}

    # Merge relative_position_bias_table from all layer into one tensor,
    # so we can use one op for gather the relative position bias for speed up
    relative_position_bias_tables = {}

    for key in state_dict:
        value = state_dict[key]

        if "relative_position_bias_table" in key:
            # transformer.blocks.0.attn.relative_position_bias_table
            layer_idx = int(key.split(".attn.")[0].split('.')[-1])
            relative_position_bias_tables[layer_idx] = value
            continue

        if "mlp" in key:
            key_imag = "transformer." + key.replace("mlp", "mlp_imag")
            new_state_dict[key_imag] = value
        elif "norm2" in key:
            key_imag = "transformer." + key.replace("norm2", "norm2_imag")
            new_state_dict[key_imag] = value
        else:
            new_key = "transformer." + key
            new_state_dict[new_key] = value

    if len(relative_position_bias_tables) > 0:
        tensor_list = []
        for layer_idx in sorted(relative_position_bias_tables.keys()):
            tensor_list.append(relative_position_bias_tables[layer_idx])
        relative_position_bias_table = torch.cat(tensor_list, dim=1)

        num_distence, _ = relative_position_bias_table.shape
        all_relative_position_bias_table = module.relative_position_bias_table.data.clone()
        all_relative_position_bias_table[:num_distence, :] = relative_position_bias_table

        new_state_dict["relative_position_bias_table"] = all_relative_position_bias_table

    return new_state_dict


def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size != new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d' % (orig_size ** 2, new_size ** 2))

        return new_pos_embed
    else:
        return pos_embed_checkpoint


def convert_deepspeed_ckpt(state_dict):
    new_state_dict = {}
    for key in state_dict:
        if key.startswith("module."):
            new_key = key[len("module."):]
            value = state_dict[key]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = state_dict[key]

    return new_state_dict


class VLMo(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.count_kdd = 0
        self.wrong_pred = 0
        self.jump = 0
        self.acc_list = []
        self.save_hyperparameters()

        self.count = 0
        self.correct = 0
        self.right_list_vlmo = []
        f = open('right_part.txt', 'r')
        a = list(f)
        f.close()
        f = open('right_part_after.txt', 'r')
        a1 = list(f)
        f.close()
        self.tokenizer_mlm = BertTokenizer.from_pretrained("bert-base-uncased",
                                                           do_lower_case="uncased" in "bert-base-uncased")
        config_atk = BertConfig.from_pretrained('bert-base-uncased')
        self.mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config_atk).to(self.device)

        def _tokenize(self, seq, tokenizer):
            seq = seq.replace('\n', '').lower()
            words = seq.split(' ')

            sub_words = []
            keys = []
            index = 0
            for word in words:
                sub = tokenizer.tokenize(word)
                sub_words += sub
                keys.append([index, index + len(sub)])
                index += len(sub)
            return words, sub_words, keys

        self.right_list = [int(l.strip('\n')) for l in a] + [int(l.strip('\n')) for l in a1]
        self.adv_store_img_source = 'attack_dir_VLMO_BASE'  # store adversarial image
        self.adv_store_txt_source = 'adv_txt_dict_VLMO_BASE.txt'  # store adversarial text
        self.adv_txt_dict = {}
        if not os.path.exists(self.adv_store_img_source):
            os.makedirs(self.adv_store_img_source)

        with open('chatgpt_all_5k.txt', 'r') as f:
            self.chatgpt = json.load(f)
        with open('chatgpt_all_5k_after.txt', 'r') as f:
            self.chatgpt.update(json.load(f))
        with open('vilt_ans_table_for_chatgpt.txt', 'r') as f:
            self.vilt_ans_table = json.load(f)
        with open('vilt_ans_table_for_chatgpt_after.txt', 'r') as f:
            self.vilt_ans_table.update(json.load(f))
        with open('vlmo_ans_table.txt', 'r') as f:
            self.vlmo_ans_table = json.load(f)
        with open('vlmo_ans_table_after.txt', 'r') as f:
            self.vlmo_ans_table.update(json.load(f))
        with open('all_correct_ans.txt', 'r') as f:
            self.all_correct_ans = json.load(f)
        with open('all_correct_ans_after.txt', 'r') as f:
            self.all_correct_ans.update(json.load(f))
        # print(len(self.right_list),len(self.vilt_ans_table.keys()))
        # exit()

        # backbone & patch projection
        self.img_size = config["image_size"]
        self.transformer = create_model(
            config["model_arch"],
            img_size=self.img_size,
            pretrained=False,
            drop_rate=0,
            drop_path_rate=config["drop_path_rate"],
            attn_drop_rate=0,
            drop_block_rate=None,
            config=self.hparams.config,
        )
        self.patch_size = self.transformer.patch_size
        self.vlffn_start_layer_index = self.transformer.vlffn_start_layer_index
        self.num_layers = len(self.transformer.blocks)
        self.num_features = self.transformer.num_features
        self.build_relative_position_embed(config)

        # language embedding
        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=self.num_features,
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_path_rate"],
            position_embedding_type="rel_pos" if self.transformer.need_relative_position_embed else "absolute",
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, self.num_features)
        self.token_type_embeddings.apply(objectives.init_weights)

        # task layers
        self.pooler = heads.Pooler(self.num_features)
        self.pooler.apply(objectives.init_weights)

        ## language modeling
        # print(config["loss_names"])
        # exit()
        if config["loss_names"]["mlm"] > 0 or config["loss_names"]["textmlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        ## image-text matching (global hard negative)
        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(self.num_features)
            self.itm_score.apply(objectives.init_weights)

        ## contrastive loss (or sampling for global hard negative)
        if config["loss_names"]["itc"] > 0:
            self.itc_text_proj = heads.ITCHead(self.num_features)
            self.itc_image_proj = heads.ITCHead(self.num_features)
            self.itc_text_proj.apply(objectives.init_weights)
            self.itc_image_proj.apply(objectives.init_weights)

            self.itc_vl_text_proj = heads.ITCHead(self.num_features)
            self.itc_vl_image_proj = heads.ITCHead(self.num_features)
            self.itc_vl_text_proj.apply(objectives.init_weights)
            self.itc_vl_image_proj.apply(objectives.init_weights)

            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.logit_vl_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.vqa_path=config['load_path']
        self.pretrain_path=config['pretrain_path']

        ## retrieval task ft
        if config["loss_names"]["irtr"] > 0:
            self.itc_text_proj = heads.ITCHead(self.num_features)
            self.itc_image_proj = heads.ITCHead(self.num_features)
            self.itc_text_proj.apply(objectives.init_weights)
            self.itc_image_proj.apply(objectives.init_weights)

            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.load_pretrained_weight()
        print('aaaaaaabb')
        # exit()

        # ===================== Downstream ===================== #
        ## VQAv2
        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(self.num_features, self.num_features * 2),
                nn.LayerNorm(self.num_features * 2),
                nn.GELU(),
                nn.Linear(self.num_features * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        ## NLVR2 (Visual reasoning)
        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(self.num_features * 2, self.num_features * 2),
                nn.LayerNorm(self.num_features * 2),
                nn.GELU(),
                nn.Linear(self.num_features * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, self.num_features)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        vlmo_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================
        print('aaaaaaabb',self.hparams.config["load_path"])
        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            rank_zero_info("Load ckpt from: {}".format(self.hparams.config["load_path"]))
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = None
            for state_dict_key in ("state_dict", "module", "model"):
                if state_dict_key in ckpt:
                    print('ccccccccc')
                    rank_zero_info("Read state dict from ckpt[%s]. " % state_dict_key)
                    state_dict = ckpt[state_dict_key]
                    break
            if state_dict_key == "module":
                print('aaaaaaa')
                state_dict = convert_deepspeed_ckpt(state_dict)
            if state_dict is None:
                print('bbbbbbbb')
                # rank_zero_info("Read state dict from ckpt. ")
                state_dict = ckpt
            print('ddddddddd')
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

            # GPU.showUtilization()
            self.reload_pretrain()
            print('reload succesuss')
            self.attack_dict = {}
            with tf.device('cpu'):
                self.USE_model = hub.load('/tfhub_pretrained/universal-sentence-encoder-large_5')

    def reload_pretrain(self):
        config = {'exp_name': 'mlm_itm_itc_base', 'seed': 1, 'datasets': ['vqa'],
                  'loss_names': {'itm': 0, 'itc': 0, 'mlm': 1, 'textmlm': 0, 'vqa': 0, 'nlvr2': 0, 'irtr': 0},
                  'batch_size': 1024, 'train_transform_keys': ['square_transform_randaug'],
                  'val_transform_keys': ['square_transform'], 'image_size': 480, 'draw_false_image': 0,
                  'image_only': False, 'text_only': False, 'vqav2_label_size': 3129, 'max_text_len': 40,
                  'max_text_len_of_initckpt': 196, 'tokenizer': 'bert-base-uncased', 'vocab_size': 30522,
                  'whole_word_masking': True, 'mlm_prob': 0.15, 'draw_false_text': 0, 'model_arch': 'vlmo_base_patch16',
                  'drop_path_rate': 0.1, 'optim_type': 'adamw', 'learning_rate': 0.0002, 'weight_decay': 0.01,
                  'decay_power': 1, 'max_epoch': 100, 'max_steps': 200000, 'warmup_steps': 0.1, 'end_lr': 0,
                  'lr_mult': 1, 'get_recall_metric': False, 'get_recall_rerank_metric': False, 'k_test': 32,
                  'resume_from': None, 'fast_dev_run': False, 'val_check_interval': 1.0, 'test_only': True,
                  'use_sharded_training': False, 'resume_during_training': False,
                  'log_dir': 'result', 'per_gpu_batchsize': 1,
                  'num_gpus': 1, 'num_nodes': 1, 'load_path': self.pretrain_path,
                  'num_workers': 8, 'precision': 32}
        # _config = copy.deepcopy(_config)

        # backbone & patch projection
        # print('start1')
        self.img_size = config["image_size"]
        self.transformer = create_model(
            config["model_arch"],
            img_size=self.img_size,
            pretrained=False,
            drop_rate=0,
            drop_path_rate=config["drop_path_rate"],
            attn_drop_rate=0,
            drop_block_rate=None,
            config=config,
        )
        # print('start2')
        self.patch_size = self.transformer.patch_size
        self.vlffn_start_layer_index = self.transformer.vlffn_start_layer_index
        self.num_layers = len(self.transformer.blocks)
        self.num_features = self.transformer.num_features
        self.build_relative_position_embed(config)

        # language embedding
        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=self.num_features,
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_path_rate"],
            position_embedding_type="rel_pos" if self.transformer.need_relative_position_embed else "absolute",
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, self.num_features)
        self.token_type_embeddings.apply(objectives.init_weights)

        # task layers
        self.pooler = heads.Pooler(self.num_features)
        self.pooler.apply(objectives.init_weights)

        ## language modeling
        if config["loss_names"]["mlm"] > 0 or config["loss_names"]["textmlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        ## image-text matching (global hard negative)
        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(self.num_features)
            self.itm_score.apply(objectives.init_weights)

        ## contrastive loss (or sampling for global hard negative)
        if config["loss_names"]["itc"] > 0:
            self.itc_text_proj = heads.ITCHead(self.num_features)
            self.itc_image_proj = heads.ITCHead(self.num_features)
            self.itc_text_proj.apply(objectives.init_weights)
            self.itc_image_proj.apply(objectives.init_weights)

            self.itc_vl_text_proj = heads.ITCHead(self.num_features)
            self.itc_vl_image_proj = heads.ITCHead(self.num_features)
            self.itc_vl_text_proj.apply(objectives.init_weights)
            self.itc_vl_image_proj.apply(objectives.init_weights)

            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.logit_vl_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        ## retrieval task ft
        if config["loss_names"]["irtr"] > 0:
            self.itc_text_proj = heads.ITCHead(self.num_features)
            self.itc_image_proj = heads.ITCHead(self.num_features)
            self.itc_text_proj.apply(objectives.init_weights)
            self.itc_image_proj.apply(objectives.init_weights)

            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # print('start')
        self.load_pretrained_weight_pretrain(config)

    def reload_vqa(self):
        config = {'exp_name': 'finetune_vqa_base_image480', 'seed': 1, 'datasets': ['vqa'],
                  'loss_names': {'itm': 0, 'itc': 0, 'mlm': 0, 'textmlm': 0, 'vqa': 1, 'nlvr2': 0, 'irtr': 0},
                  'batch_size': 128, 'train_transform_keys': ['square_transform_randaug'],
                  'val_transform_keys': ['square_transform'], 'image_size':480, 'draw_false_image': 0,
                  'image_only': False, 'text_only': False, 'vqav2_label_size': 3129, 'max_text_len': 40,
                  'max_text_len_of_initckpt': 196, 'tokenizer': 'bert-base-uncased', 'vocab_size': 30522,
                  'whole_word_masking': False, 'mlm_prob': 0.15, 'draw_false_text': 0,
                  'model_arch': 'vlmo_base_patch16', 'drop_path_rate': 0.15, 'optim_type': 'adamw',
                  'learning_rate': 3e-05, 'weight_decay': 0.01, 'decay_power': 1, 'max_epoch': 10, 'max_steps': None,
                  'warmup_steps': 0.1, 'end_lr': 0, 'lr_mult': 20, 'get_recall_metric': False,
                  'get_recall_rerank_metric': False, 'k_test': 32, 'resume_from': None, 'fast_dev_run': False,
                  'val_check_interval': 1.0, 'test_only': True, 'use_sharded_training': False,
                  'resume_during_training': False, 'log_dir': 'result',
                  'per_gpu_batchsize': 1, 'num_gpus': 1, 'num_nodes': 1,
                  'load_path': self.vqa_path, 'num_workers': 8, 'precision': 32}
        # backbone & patch projection
        self.img_size = config["image_size"]
        self.transformer = create_model(
            config["model_arch"],
            img_size=self.img_size,
            pretrained=False,
            drop_rate=0,
            drop_path_rate=config["drop_path_rate"],
            attn_drop_rate=0,
            drop_block_rate=None,
            config=config,
        )
        self.patch_size = self.transformer.patch_size
        self.vlffn_start_layer_index = self.transformer.vlffn_start_layer_index
        self.num_layers = len(self.transformer.blocks)
        self.num_features = self.transformer.num_features
        self.build_relative_position_embed(config)

        # language embedding
        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=self.num_features,
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_path_rate"],
            position_embedding_type="rel_pos" if self.transformer.need_relative_position_embed else "absolute",
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, self.num_features)
        self.token_type_embeddings.apply(objectives.init_weights)

        # task layers
        self.pooler = heads.Pooler(self.num_features)
        self.pooler.apply(objectives.init_weights)

        ## language modeling
        if config["loss_names"]["mlm"] > 0 or config["loss_names"]["textmlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        ## image-text matching (global hard negative)
        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(self.num_features)
            self.itm_score.apply(objectives.init_weights)

        ## contrastive loss (or sampling for global hard negative)
        if config["loss_names"]["itc"] > 0:
            self.itc_text_proj = heads.ITCHead(self.num_features)
            self.itc_image_proj = heads.ITCHead(self.num_features)
            self.itc_text_proj.apply(objectives.init_weights)
            self.itc_image_proj.apply(objectives.init_weights)

            self.itc_vl_text_proj = heads.ITCHead(self.num_features)
            self.itc_vl_image_proj = heads.ITCHead(self.num_features)
            self.itc_vl_text_proj.apply(objectives.init_weights)
            self.itc_vl_image_proj.apply(objectives.init_weights)

            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.logit_vl_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        ## retrieval task ft
        if config["loss_names"]["irtr"] > 0:
            self.itc_text_proj = heads.ITCHead(self.num_features)
            self.itc_image_proj = heads.ITCHead(self.num_features)
            self.itc_text_proj.apply(objectives.init_weights)
            self.itc_image_proj.apply(objectives.init_weights)

            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.load_pretrained_weight_pretrain(config)

        # print(self.relative_position_index,self.text_relative_position_index,self.text_imag_relative_position_index)

        # ===================== Downstream ===================== #
        # VQAv2
        if config["loss_names"]["vqa"] > 0:
            vs = config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(self.num_features, self.num_features * 2),
                nn.LayerNorm(self.num_features * 2),
                nn.GELU(),
                nn.Linear(self.num_features * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        ## NLVR2 (Visual reasoning)
        if config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(self.num_features * 2, self.num_features * 2),
                nn.LayerNorm(self.num_features * 2),
                nn.GELU(),
                nn.Linear(self.num_features * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, self.num_features)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        vlmo_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================
        # print(self.hparams.config["load_path"]=config["load_path"]
        if config["load_path"] != "" and config["test_only"]:
            # rank_zero_info("Load ckpt from: {}".format(config["load_path"]))
            ckpt = torch.load(config["load_path"], map_location="cpu")

            state_dict = None

            for state_dict_key in ("state_dict", "module", "model"):
                if state_dict_key in ckpt:
                    print('ccccccccc')
                    rank_zero_info("Read state dict from ckpt[%s]. " % state_dict_key)
                    state_dict = ckpt[state_dict_key]
                    break
            if state_dict_key == "module":
                print('aaaaaaa')
                state_dict = convert_deepspeed_ckpt(state_dict)
            if state_dict is None:
                # print('bbbbbbbb')
                # rank_zero_info("Read state dict from ckpt. ")
                state_dict = ckpt
            # print('ddddddddd')
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

    def load_pretrained_weight(self):
        if self.hparams.config["load_path"] != "" and not self.hparams.config["test_only"]:
            print('loaddddddd!')
            config = self.hparams.config
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            rank_zero_info("Load ckpt from: {}".format(self.hparams.config["load_path"]))

            state_dict = None

            for state_dict_key in ("state_dict", "module", "model"):
                if state_dict_key in ckpt:
                    rank_zero_info("Read state dict from ckpt[%s]. " % state_dict_key)
                    state_dict = ckpt[state_dict_key]
                    break
            if state_dict_key == "module":
                state_dict = convert_deepspeed_ckpt(state_dict)
            if state_dict is None:
                rank_zero_info("Read state dict from ckpt. ")
                state_dict = ckpt

            for key in state_dict:
                var = state_dict[key]
                rank_zero_info("%s = %s" % (key, str(var.size())))

            rank_zero_info(config["loss_names"])
            if config["loss_names"]["textmlm"] > 0:
                rank_zero_info("convert to textpt")
                state_dict = convert_to_textpt_ckpt(state_dict, self)

            max_text_len = config["max_text_len"]
            # print('here')
            # exit()
            if "text_embeddings.position_embeddings.weight" in state_dict and state_dict[
                "text_embeddings.position_embeddings.weight"].size(0) != max_text_len:
                state_dict["text_embeddings.position_embeddings.weight"].data = state_dict[
                                                                                    "text_embeddings.position_embeddings.weight"].data[
                                                                                :max_text_len, :]
                state_dict["text_embeddings.position_ids"].data = state_dict["text_embeddings.position_ids"].data[:,
                                                                  :max_text_len]
                rank_zero_info("text position_embeddings size: {}".format(
                    state_dict["text_embeddings.position_embeddings.weight"].size()))
                for check_key in (
                "relative_position_index", "text_relative_position_index", "text_imag_relative_position_index"):
                    if check_key in state_dict:
                        state_dict.pop(check_key)

            if "transformer.pos_embed" in state_dict:
                print('have')
                exit()
                pos_embed_reshaped = interpolate_pos_embed(state_dict['transformer.pos_embed'], self.transformer)
                state_dict['transformer.pos_embed'] = pos_embed_reshaped
            print('aaaaxxxxx')
            # exit()
            if "relative_position_bias_table" in state_dict:
                rel_pos_bias = state_dict["relative_position_bias_table"]
                print(rel_pos_bias.shape)
                src_num_pos, num_attn_heads = rel_pos_bias.size()
                dst_num_pos, _ = self.relative_position_bias_table.size()
                dst_patch_shape = self.transformer.patch_embed.patch_shape
                if dst_patch_shape[0] != dst_patch_shape[1]:
                    raise NotImplementedError()
                num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
                src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
                dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
                if src_size != dst_size:
                    state_dict.pop("relative_position_index")
                    state_dict.pop("text_relative_position_index")
                    state_dict.pop("text_imag_relative_position_index")

                    rank_zero_info("Position interpolate from %dx%d to %dx%d" % (
                        src_size, src_size, dst_size, dst_size))
                    extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                    rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    rank_zero_info("Original positions = %s" % str(x))
                    rank_zero_info("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(num_attn_heads):
                        z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                        f = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(
                            torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                    rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                    new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                    state_dict["relative_position_bias_table"] = new_rel_pos_bias
                    # print(new_rel_pos_bias.shape)
                    # exit()

            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
    def load_pretrained_weight_pretrain(self, config):
        ckpt = torch.load(config["load_path"], map_location="cpu")
        state_dict = None

        for state_dict_key in ("state_dict", "module", "model"):
            if state_dict_key in ckpt:
                print('have-4')
                # exit()
                # rank_zero_info("Read state dict from ckpt[%s]. " % state_dict_key)
                state_dict = ckpt[state_dict_key]
                break
        if state_dict_key == "module":
            print('have-1')
            # exit()
            state_dict = convert_deepspeed_ckpt(state_dict)
        if state_dict is None:
            state_dict = ckpt

        for key in state_dict:
            var = state_dict[key]
        if config["loss_names"]["textmlm"] > 0:
            print('have-0')
            # exit()
            # rank_zero_info("convert to textpt")
            state_dict = convert_to_textpt_ckpt(state_dict, self)

        max_text_len = config["max_text_len"]
        if "text_embeddings.position_embeddings.weight" in state_dict and state_dict[
            "text_embeddings.position_embeddings.weight"].size(0) != max_text_len:
            print('have-5')
            # exit()
            state_dict["text_embeddings.position_embeddings.weight"].data = state_dict[
                                                                                "text_embeddings.position_embeddings.weight"].data[
                                                                            :max_text_len, :]
            state_dict["text_embeddings.position_ids"].data = state_dict["text_embeddings.position_ids"].data[:,
                                                              :max_text_len]
            rank_zero_info("text position_embeddings size: {}".format(
                state_dict["text_embeddings.position_embeddings.weight"].size()))
            for check_key in (
                    "relative_position_index", "text_relative_position_index", "text_imag_relative_position_index"):
                if check_key in state_dict:
                    print('have-6')
                    # exit()
                    print("1111111")
                    state_dict.pop(check_key)
        # print(state_dict["transformer.pos_embed"])
        if "transformer.pos_embed" in state_dict:
            print('have-7')
            # exit()
            pos_embed_reshaped = interpolate_pos_embed(state_dict['transformer.pos_embed'], self.transformer)
            state_dict['transformer.pos_embed'] = pos_embed_reshaped

        if "relative_position_bias_table" in state_dict:
            # print('have-8')
            # exit()
            rel_pos_bias = state_dict["relative_position_bias_table"]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = self.relative_position_bias_table.size()
            dst_patch_shape = self.transformer.patch_embed.patch_shape
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                state_dict.pop("relative_position_index")
                state_dict.pop("text_relative_position_index")
                state_dict.pop("text_imag_relative_position_index")

                # rank_zero_info("Position interpolate from %dx%d to %dx%d" % (
                #     src_size, src_size, dst_size, dst_size))
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                # rank_zero_info("Original positions = %s" % str(x))
                # rank_zero_info("Target positions = %s" % str(dx))

                all_rel_pos_bias = []

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                    f = interpolate.interp2d(x, y, z, kind='cubic')
                    all_rel_pos_bias.append(
                        torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                state_dict["relative_position_bias_table"] = new_rel_pos_bias
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

    def get_rel_pos_bias(self, relative_position_index):
        if self.relative_position_embed:
            relative_position_bias = F.embedding(
                relative_position_index.long().to(self.relative_position_bias_table.device),
                self.relative_position_bias_table)
            all_relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, x, y
            relative_position_bias_list = torch.chunk(all_relative_position_bias, self.num_layers, dim=0)
            return relative_position_bias_list
        else:
            return [None] * self.num_layers

    def build_relative_position_embed(self, config):
        if not self.transformer.need_relative_position_embed:
            self.relative_position_embed = False
            self.text_imag_relative_position_index = None
            self.text_relative_position_index = None
            self.relative_position_index = None
            return
        self.relative_position_embed = True
        window_size = (int(self.img_size / self.patch_size), int(self.img_size / self.patch_size))  # (14, 14)
        # rank_zero_info("window_size: {}".format(window_size))
        num_heads = self.transformer.num_heads
        max_text_len_of_initckpt = config["max_text_len_of_initckpt"]  # 196
        max_text_len = config["max_text_len"]  # 40
        max_imag_len = window_size[0] * window_size[1] + 1  # 197
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.text_num_relative_distance = 2 * max_text_len_of_initckpt
        self.all_num_relative_distance = self.num_relative_distance + self.text_num_relative_distance + 2

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.all_num_relative_distance, num_heads * self.num_layers))

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1
        self.relative_position_index = relative_position_index

        text_position_ids = torch.arange(max_text_len - 1)
        text_rel_pos_mat = text_position_ids.unsqueeze(-2) - text_position_ids.unsqueeze(-1)
        min_distance = int(2 - max_text_len_of_initckpt)  # -194
        # rank_zero_info("min_distance: {}".format(min_distance))
        text_rel_pos_mat = text_rel_pos_mat - min_distance
        text_rel_pos_mat += (self.num_relative_distance + 2)
        text_relative_position_index = \
            torch.zeros(size=(max_text_len,) * 2, dtype=relative_coords.dtype)
        text_relative_position_index[1:, 1:] = text_rel_pos_mat
        text_relative_position_index[0, 0:] = self.all_num_relative_distance - 3
        text_relative_position_index[0:, 0] = self.all_num_relative_distance - 2
        text_relative_position_index[0, 0] = self.all_num_relative_distance - 1
        self.text_relative_position_index = text_relative_position_index

        text2imag_relative_position_index = torch.ones(max_text_len, max_imag_len) * (self.num_relative_distance)
        imag2text_relative_position_index = torch.ones(max_imag_len, max_text_len) * (self.num_relative_distance + 1)

        text_row_relative_position_index = torch.cat((text_relative_position_index, text2imag_relative_position_index),
                                                     1)
        imag_row_relative_position_index = torch.cat((imag2text_relative_position_index, relative_position_index), 1)
        text_imag_relative_position_index = torch.cat(
            (text_row_relative_position_index, imag_row_relative_position_index), 0)
        self.text_imag_relative_position_index = text_imag_relative_position_index
        # print(self.text_imag_relative_position_index.shape)
        # exit()

    def infer(
            self,
            batch,
            mask_text=False,
            mask_image=False,
            image_token_type_idx=1,
            image_embeds=None,
            image_masks=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        # print('device',text_ids.device,self.device,self.relative_position_index.device,self.transformer.blocks[6].mlp_imag.fc1.weight.device)
        text_embeds = self.text_embeddings(text_ids)
        # print('bbb', text_ids, text_masks)

        img = batch[imgkey][0]

        image_embeds, image_masks = self.transformer.visual_embed(img)
        image_masks = image_masks.long().to(device=img.get_device())
        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )
        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        x = co_embeds
        relative_position_bias_list = self.get_rel_pos_bias(self.text_imag_relative_position_index)

        # attn_list = []
        feats_list = []
        feats_list.append(co_embeds)
        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=co_masks, modality_type="vl", relative_position_bias=relative_position_bias_list[i])
            feats_list.append(x)

        x = self.transformer.norm(x)
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1]:],
        )
        cls_feats = self.pooler(x)
        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image": img,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "feats": feats_list,
        }

        return ret

    def infer_text(
            self,
            batch,
            mask_text=False,
    ):
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))

        co_embeds = text_embeds
        co_masks = text_masks

        x = co_embeds
        all_hidden_states = []
        relative_position_bias_list = self.get_rel_pos_bias(self.text_relative_position_index)

        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=co_masks, modality_type="text", relative_position_bias=relative_position_bias_list[i])
            all_hidden_states.append(x)

        vlffn_hiddens = all_hidden_states[self.vlffn_start_layer_index - 1]
        for vlffn_index in range(self.vlffn_start_layer_index, self.num_layers):
            vlffn_hiddens = self.transformer.blocks[vlffn_index](vlffn_hiddens, mask=co_masks, modality_type="vl",
                                                                 relative_position_bias=relative_position_bias_list[
                                                                     vlffn_index])

        lffn_hiddens = all_hidden_states[-1]

        lffn_hiddens = self.transformer.norm(lffn_hiddens)
        text_feats, image_feats = (
            lffn_hiddens,
            None,
        )

        cls_feats = self.itc_text_proj(lffn_hiddens[:, 0])
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        vlffn_hiddens = self.transformer.norm(vlffn_hiddens)
        cls_vlffn_feats = self.itc_vl_text_proj(vlffn_hiddens[:, 0])
        cls_vlffn_feats = cls_vlffn_feats / cls_vlffn_feats.norm(dim=-1, keepdim=True)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "cls_vlffn_feats": cls_vlffn_feats,
            "raw_cls_feats": x[:, 0],
            "image_masks": None,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
        }

        return ret

    def infer_text_ft(
            self,
            batch,
            mask_text=False,
    ):
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))

        co_embeds = text_embeds
        co_masks = text_masks

        x = co_embeds
        all_hidden_states = []
        relative_position_bias_list = self.get_rel_pos_bias(self.text_relative_position_index)

        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=co_masks, modality_type="text", relative_position_bias=relative_position_bias_list[i])
            all_hidden_states.append(x)

        lffn_hiddens = all_hidden_states[-1]

        lffn_hiddens = self.transformer.norm(lffn_hiddens)
        text_feats, image_feats = (
            lffn_hiddens,
            None,
        )

        cls_feats = self.itc_text_proj(lffn_hiddens[:, 0])
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "cls_vlffn_feats": None,
            "raw_cls_feats": x[:, 0],
            "image_masks": None,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
        }

        return ret

    def infer_text_mlm(
            self,
            batch,
            mask_text=False,
    ):
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))

        co_embeds = text_embeds
        co_masks = text_masks

        x = co_embeds
        all_hidden_states = []
        relative_position_bias_list = self.get_rel_pos_bias(self.text_relative_position_index)

        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=co_masks, modality_type="text", relative_position_bias=relative_position_bias_list[i])
            all_hidden_states.append(x)

        lffn_hiddens = all_hidden_states[-1]

        lffn_hiddens = self.transformer.norm(lffn_hiddens)
        text_feats, image_feats = (
            lffn_hiddens,
            None,
        )

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": None,
            "cls_vlffn_feats": None,
            "raw_cls_feats": x[:, 0],
            "image_masks": None,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
        }

        return ret

    def infer_image(
            self,
            batch,
            mask_image=False,
            image_token_type_idx=1,
            image_embeds=None,
            image_masks=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        img = batch[imgkey][0]
        image_embeds, image_masks = self.transformer.visual_embed(img)

        image_masks = image_masks.long().to(device=img.get_device())
        image_embeds = image_embeds + self.token_type_embeddings(
            torch.full_like(image_masks, image_token_type_idx)
        )

        co_embeds = image_embeds
        co_masks = image_masks

        x = co_embeds
        all_hidden_states = []
        relative_position_bias_list = self.get_rel_pos_bias(self.relative_position_index)

        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=co_masks, modality_type="image", relative_position_bias=relative_position_bias_list[i])
            all_hidden_states.append(x)

        vlffn_hiddens = all_hidden_states[self.vlffn_start_layer_index - 1]
        for vlffn_index in range(self.vlffn_start_layer_index, self.num_layers):
            vlffn_hiddens = self.transformer.blocks[vlffn_index](vlffn_hiddens, mask=co_masks, modality_type="vl",
                                                                 relative_position_bias=relative_position_bias_list[
                                                                     vlffn_index])

        vffn_hiddens = all_hidden_states[-1]

        vffn_hiddens = self.transformer.norm(vffn_hiddens)
        text_feats, image_feats = (
            None,
            vffn_hiddens,
        )

        cls_feats = self.itc_image_proj(vffn_hiddens[:, 0])
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        vlffn_hiddens = self.transformer.norm(vlffn_hiddens)
        cls_vlffn_feats = self.itc_vl_image_proj(vlffn_hiddens[:, 0])
        cls_vlffn_feats = cls_vlffn_feats / cls_vlffn_feats.norm(dim=-1, keepdim=True)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "cls_vlffn_feats": cls_vlffn_feats,
            "raw_cls_feats": x[:, 0],
            "image_masks": image_masks,
            "text_labels": None,
            "text_ids": None,
            "text_masks": None,
        }

        return ret

    def infer_image_ft(
            self,
            batch,
            mask_image=False,
            image_token_type_idx=1,
            image_embeds=None,
            image_masks=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        img = batch[imgkey][0]
        image_embeds, image_masks = self.transformer.visual_embed(img)

        image_masks = image_masks.long().to(device=img.get_device())
        image_embeds = image_embeds + self.token_type_embeddings(
            torch.full_like(image_masks, image_token_type_idx)
        )

        co_embeds = image_embeds
        co_masks = image_masks

        x = co_embeds
        all_hidden_states = []
        relative_position_bias_list = self.get_rel_pos_bias(self.relative_position_index)

        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=co_masks, modality_type="image", relative_position_bias=relative_position_bias_list[i])
            all_hidden_states.append(x)

        vffn_hiddens = all_hidden_states[-1]

        vffn_hiddens = self.transformer.norm(vffn_hiddens)
        text_feats, image_feats = (
            None,
            vffn_hiddens,
        )

        cls_feats = self.itc_image_proj(vffn_hiddens[:, 0])
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "cls_vlffn_feats": None,
            "raw_cls_feats": x[:, 0],
            "image_masks": image_masks,
            "text_labels": None,
            "text_ids": None,
            "text_masks": None,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        # print('tasks',self.current_tasks)
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Textonly Masked Language Modeling
        if "textmlm" in self.current_tasks:
            ret.update(objectives.compute_textonly_mlm(self, batch))

        # Contrastive loss for pretraining
        if "itc" in self.current_tasks:
            ret.update(objectives.compute_itc(self, batch))

        # Contrastive loss for finetuning
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        # Image Text Matching with global hard negative, must use with itc
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_hardneg(self, batch, ret["itc_i2t_logits"], ret["itc_t2i_logits"]))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))
        # print('return here')

        return ret

    def training_step(self, batch, batch_idx):
        vlmo_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        vlmo_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vlmo_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vlmo_utils.epoch_wrapup(self)

    def filter(self, ori_words):
        stop_words = ['on', 'and', 'in', 'his', 'her', 'its']
        for i in stop_words:
            if i in ori_words:
                ori_words.remove(i)
        return ori_words

    def Gen_ori_feats(self, batch):
        # if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
        #     ckpt = torch.load('pretrain_model/vilt_200k_mlm_itm.ckpt', map_location="cpu")
        #     state_dict = ckpt["state_dict"]
        #     self.load_state_dict(state_dict, strict=False)
        # vilt_utils.set_task(self)
        # print('hhhhhhh1')
        output = self(batch)
        # print('hhhhhhh2')
        # print(self(batch))
        # exit()
        # print('output_keys',output.keys())
        # exit()
        tgt_feats = torch.stack(output['feats_list'],
                                axis=1).detach().cpu().cuda()
        # print('tgt',tgt_feats.shape)

        tgt_feats = tgt_feats[0, :, 0, :]
        target_feats = torch.stack(output['feats_list'],
                                   axis=1).detach().cpu().cuda()
        feats_list_img = target_feats[0, :, 40:]
        text_masks = torch.where(output['text_masks'][0] == 1)
        feats_list_text = target_feats[0, :, text_masks[0]]
        feats_list = torch.cat([feats_list_text, feats_list_img], axis=1)

        return tgt_feats, feats_list, feats_list_img

    def _tokenize(self, seq, tokenizer):
        seq = seq.replace('\n', '').lower()
        words = seq.split(' ')

        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)
        return words, sub_words, keys

    def pgd_attack_vl(self, x):
        image_token_type_idx = 1
        mask_image = False
        image_embeds = None
        image_masks = None
        mask_text = False
        do_mlm = "_mlm" if mask_text else ""

        # text_ids = self.batch[f"text_ids{do_mlm}"]
        # text_embeds = self.text_embeddings(text_ids)

        text_embeds = x[1]
        text_masks = self.batch[f"text_masks"]
        # print('11',text_ids)
        # exit()

        image_embeds, image_masks = self.transformer.visual_embed(x[0])
        image_masks = image_masks.long().to(device=x[0].get_device())

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)  # modal type embedding
            ),
        )

        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        xx = co_embeds
        relative_position_bias_list = self.get_rel_pos_bias(self.text_imag_relative_position_index)

        # attn_list = []
        feats = []

        # value_list = []
        feats.append(co_embeds)
        for i, blk in enumerate(self.transformer.blocks):
            xx = blk(xx, mask=co_masks, modality_type="vl", relative_position_bias=relative_position_bias_list[i])
            feats.append(xx)

        xx = self.transformer.norm(xx)
        text_feats, image_feats = (
            xx[:, : text_embeds.shape[1]],
            xx[:, text_embeds.shape[1]:],
        )
        # cls_weight_dict = self.cls_value[str(self.batch['qid'][0])]
        cls_feats = self.pooler(xx)
        target_feats = torch.stack(feats, axis=1)
        # target_feats = torch.stack([feats[int(i)] for i in cls_weight_dict.keys()],axis=1)
        feats_list_img = target_feats[0, :, 40:]
        text_masks = torch.where(text_masks[0] == 1)
        feats_list_text = target_feats[0, :, text_masks[0]]
        feats_list = torch.cat([feats_list_text, feats_list_img], axis=1)
        six_layer_feats = torch.stack(feats, axis=1)  # .detach().cpu().cuda()
        six_layer_feats = six_layer_feats[0, :, 0, :]
        return [cls_feats, six_layer_feats, feats_list]

    def pgd_attack(self, x):
        image_token_type_idx = 1
        mask_image = False
        image_embeds = None
        image_masks = None
        mask_text = False
        do_mlm = "_mlm" if mask_text else ""

        text_ids = self.batch[f"text_ids{do_mlm}"]
        text_masks = self.batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)
        # print()
        # print('11',text_ids)
        # exit()

        image_embeds, image_masks = self.transformer.visual_embed(x)
        image_masks = image_masks.long().to(device=x.get_device())

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)  # modal type embedding
            ),
        )

        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        x = co_embeds
        relative_position_bias_list = self.get_rel_pos_bias(self.text_imag_relative_position_index)

        # attn_list = []
        feats = []
        # value_list = []
        feats.append(co_embeds)
        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=co_masks, modality_type="vl", relative_position_bias=relative_position_bias_list[i])
            feats.append(x)
        x = self.transformer.norm(x)
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1]:],
        )
        # print('text_embeds',text_embeds.shape)
        # exit()
        # cls_weight_dict = self.cls_value[str(self.batch['qid'][0])]
        cls_feats = self.pooler(x)
        target_feats = torch.stack(feats, axis=1)
        # print('target',target_feats.shape)
        # target_feats = torch.stack([feats[int(i)] for i in cls_weight_dict.keys()],axis=1)
        feats_list_img = target_feats[0, :, 40:]
        text_masks = torch.where(text_masks[0] == 1)
        feats_list_text = target_feats[0, :, text_masks[0]]

        feats_list = torch.cat([feats_list_text, feats_list_img], axis=1)
        six_layer_feats = torch.stack(feats, axis=1)  # .detach().cpu().cuda()
        six_layer_feats = six_layer_feats[0, :, 0, :]
        # print('fshape',six_layer_feats.shape,feats_list.shape)
        return [cls_feats, six_layer_feats, feats_list]

    def pgd_mlm_attack(self, x):
        image_token_type_idx = 1
        mask_image = False
        image_embeds = None
        image_masks = None
        mask_text = True
        do_mlm = "_mlm" if mask_text else ""
        # print(self.batch[f"text_ids{do_mlm}"])
        # exit()
        text_ids = self.batch[f"text_ids{do_mlm}"]
        text_masks = self.batch[f"text_mask_mlm"]
        text_embeds = self.text_embeddings(text_ids)
        # print('aaa',text_ids,text_masks)
        # exit()

        # if image_embeds is None and image_masks is None:
        #     (
        #         image_embeds,
        #         image_masks,
        #         patch_index,
        #         image_labels,
        #         _,
        #     ) = self.transformer.visual_embed(
        #         x,
        #         max_image_len=self.hparams.config["max_image_len"],
        #         mask_it=mask_image,
        #     )
        # else:
        #     patch_index, image_labels = (
        #         None,
        #         None,
        #     )
        image_embeds, image_masks = self.transformer.visual_embed(x)
        image_masks = image_masks.long().to(device=x.get_device())

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )
        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)
        x = co_embeds
        # attn_list = []
        # value_list = []
        relative_position_bias_list = self.get_rel_pos_bias(self.text_imag_relative_position_index)

        # attn_list = []
        feats = []
        # value_list = []
        feats.append(co_embeds)
        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=co_masks, modality_type="vl", relative_position_bias=relative_position_bias_list[i])
            feats.append(x)
        x = self.transformer.norm(x)
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1]:],
        )
        # id2answer = (
        #     self.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
        #     if "vqa_trainval" in self.trainer.datamodule.dm_dicts
        #     else self.trainer.datamodule.dm_dicts["vqa"].id2answer
        # )
        cls_feats = self.pooler(x)
        # print('cls',cls_feats)
        # vqa_logits=self.vqa_classifier(cls_feats)
        # vqa_preds = vqa_logits.argmax(dim=-1)
        # vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
        # print(vqa_preds)
        # exit()
        mlm_logits = self.mlm_score(text_feats)
        target_feats = torch.stack(feats, axis=1)
        feats_list_img = target_feats[0, :, 40:]
        text_masks = torch.where(text_masks[0] == 1)
        feats_list_text = target_feats[0, :, text_masks[0]]
        feats_list = torch.cat([feats_list_text, feats_list_img], axis=1)
        six_layer_feats = torch.stack(feats, axis=1)  # .detach().cpu().cuda()
        six_layer_feats = six_layer_feats[0, :, 0, :]
        return [mlm_logits, six_layer_feats, feats_list]

    def cal_text_attack_list(self, ori_text):
        iter_list = []
        bert_cand_list = []
        vlmo_utils.set_task(self)
        text = ori_text.lower()
        feature = Feature(text)
        tokenizer = self.tokenizer_mlm
        # words, sub_words, keys = self._tokenize(feature.seq.strip('.'), tokenizer)
        words, sub_words, keys = self._tokenize(ori_text.strip('?').lower(), self.tokenizer_mlm)
        bert_cand_list = [None for i in range(len(words))]
        # print(words,sub_words,keys)
        substitute_list = []
        for (wo, key) in zip(words, keys):
            if key[1] - key[0] == 1 and wo not in filter_words:
                substitute_list.append(key)
        count = len(substitute_list)
        if count == 0:
            return [], []
        count += 1
        # count
        if int(40 / count) % 2 == 0:
            iter_list = [int(40 / count) for i in range(count)]
            iter_list[-1] += 40 - sum(iter_list)
        else:
            iter_list = [int(40 / count) - 1 for i in range(count)]
            iter_list[-1] += 40 - sum(iter_list)
        max_length = 512
        sub_words = ['[CLS]'] + sub_words[:2] + sub_words[2:max_length - 2] + ['[SEP]']
        input_ids_ = torch.tensor([tokenizer.convert_tokens_to_ids(sub_words)])
        word_predictions = self.mlm_model(input_ids_.to(self.device))[0].squeeze()  # seq-len(sub) vocab
        word_pred_scores_all, word_predictions = torch.topk(word_predictions, 5, -1)
        word_predictions = word_predictions[1:len(sub_words) + 1, :]
        word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]
        for sub in substitute_list:
            substitutes = word_predictions[sub[0]:sub[1]]
            word_pred_scores = word_pred_scores_all[sub[0]:sub[1]]
            substitutes = self.get_substitues(substitutes, tokenizer, self.mlm_model,
                                              substitutes_score=word_pred_scores)
            for substitute in substitutes:
                if substitute == words[keys.index(sub)]:
                    continue  # filter out original word
                if '##' in substitute:
                    continue  # filter out sub-word
                if substitute in filter_words:
                    continue
                if bert_cand_list[keys.index(sub)] is None:
                    bert_cand_list[keys.index(sub)] = []
                    bert_cand_list[keys.index(sub)].append(substitute)
                else:
                    bert_cand_list[keys.index(sub)].append(substitute)
        return iter_list, bert_cand_list

    def get_bpe_substitues(self, substitutes, tokenizer, mlm_model):
        substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates
        all_substitutes = []
        for i in range(substitutes.size(0)):
            if len(all_substitutes) == 0:
                lev_i = substitutes[i]
                all_substitutes = [[int(c)] for c in lev_i]
            else:
                lev_i = []
                for all_sub in all_substitutes:
                    for j in substitutes[i]:
                        lev_i.append(all_sub + [int(j)])
                all_substitutes = lev_i
        c_loss = torch.nn.CrossEntropyLoss(reduction='none')
        word_list = []
        all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
        all_substitutes = all_substitutes[:24].to(self.device)
        N, L = all_substitutes.size()
        word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size
        ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))  # [ N*L ]
        ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
        _, word_list = torch.sort(ppl)
        word_list = [all_substitutes[i] for i in word_list]
        final_words = []
        for word in word_list:
            tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
            text = tokenizer.convert_tokens_to_string(tokens)
            final_words.append(text)
        return final_words

    def get_substitues(self, substitutes, tokenizer, mlm_model, substitutes_score=None, use_bpe=True, threshold=0.3):
        words = []
        sub_len, k = substitutes.size()  # sub-len, k

        if sub_len == 0:
            return words

        elif sub_len == 1:
            for (i, j) in zip(substitutes[0], substitutes_score[0]):
                if threshold != 0 and j < threshold:
                    break
                words.append(tokenizer._convert_id_to_token(int(i)))
        else:
            if use_bpe == 1:
                words = self.get_bpe_substitues(substitutes, tokenizer, mlm_model)
            else:
                return words
        return words

    def dir_sim(self, cand_emb_dir, attack_grad):
        # print(cand_emb_dir.shape,attack_grad.shape)

        cand_norm = F.normalize(cand_emb_dir, p=2, dim=0)
        attack_norm = F.normalize(attack_grad, p=2, dim=0)
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        dir_sim = cos(cand_norm, attack_norm)

        return dir_sim

    def update_adv_text(self, text_embed_gradient, bert_cand_list, cand_wd_list, adv_text, attack_vector, ori_emb,
                        ori_text):
        words, sub_words, keys = self._tokenize(adv_text.strip('?').lower(), self.tokenizer_mlm)
        ori_words = copy.deepcopy(words)
        adv_words = copy.deepcopy(words)
        sort_list = []
        dir_sim_list = []
        occupied_list = []
        for idx, (cand_wd_idx, sub_wd_idx) in enumerate(zip(cand_wd_list, attack_vector)):
            attack_grad = text_embed_gradient[0, idx]
            cand_list = bert_cand_list[cand_wd_idx]
            cand_words_words = copy.deepcopy(words)
            for idd, cand_wd in enumerate(cand_list):
                sort_list.append([cand_wd_idx, idd])
                if cand_wd_idx >= len(cand_words_words):
                    print('onebug', adv_text.strip('?').lower(), words, ori_text, cand_wd_list, bert_cand_list,
                          attack_vector)
                    return self.tokenizer_mlm.convert_tokens_to_string(ori_words) + '?', []
                cand_words_words[cand_wd_idx] = cand_wd
                adv_sentence = ' '.join(cand_words_words) + '?'
                encoding_adv = self.tokenizer_mlm(
                    adv_sentence,
                    padding="max_length",
                    truncation=True,
                    max_length=40,
                    return_special_tokens_mask=True,
                )
                encode_ids_adv = torch.tensor(encoding_adv["input_ids"]).unsqueeze(0).cuda()
                adv_text_embeds = self.text_embeddings(encode_ids_adv)
                cand_emb_dir = adv_text_embeds[0, sub_wd_idx] - ori_emb[0, sub_wd_idx]
                dir_sim = self.dir_sim(cand_emb_dir, attack_grad)
                dir_sim_list.append(dir_sim)
                # print('words',cand_words_words)
        ll = sorted(range(len(dir_sim_list)), key=lambda k: dir_sim_list[k], reverse=True)
        sorted_op_list = [sort_list[i] for i in ll]
        sorted_sim_list = [dir_sim_list[i] for i in ll]
        # print(sorted_op_list,dir_sim_list,sorted_sim_list)
        # exit()
        sim_threshold = 0.95
        op_mlm_list = []
        for (dir_sim, op) in zip(sorted_sim_list, sorted_op_list):
            # if dir_sim<0:
            #     continue
            if op[0] in occupied_list:
                continue
            # adv_words[op[0]]=bert_cand_list[op[0]][op[1]]
            temp_replace = copy.deepcopy(adv_words)
            temp_replace[op[0]] = bert_cand_list[op[0]][op[1]]
            temp_text = self.tokenizer_mlm.convert_tokens_to_string(temp_replace) + '?'
            # print('temp',temp_text)
            embs = self.USE_model([ori_text, temp_text]).numpy()
            norm = np.linalg.norm(embs, axis=1)
            embs = embs / norm[:, None]
            USE_sim = (embs[:1] * embs[1:]).sum(axis=1)[0]
            if USE_sim > sim_threshold:
                sim_threshold = USE_sim
                occupied_list.append(op[0])
                adv_words = temp_replace
                op_mlm_list.append([ori_words[op[0]], bert_cand_list[op[0]][op[1]]])
        # exit()
        return self.tokenizer_mlm.convert_tokens_to_string(adv_words) + '?', op_mlm_list

    def update_mlm_text(self, op_mlm_list, list_words):
        for op_mlm in op_mlm_list:
            ori_word = op_mlm[0]
            cand_word = op_mlm[1]
            if ori_word in list_words:
                index_list = [index for (index, value) in enumerate(list_words) if value == ori_word]
                for idx in index_list:
                    list_words[idx] = cand_word
        text_mlm = ' '.join(list_words) + '.'
        encoding = self.tokenizer_mlm(
            text_mlm,
            padding="max_length",
            truncation=True,
            max_length=40,
            return_special_tokens_mask=True,
        )
        self.batch[f"text_mask_mlm"] = torch.tensor(encoding["attention_mask"]).unsqueeze(0).cuda()
        self.batch[f"text_ids_mlm"] = torch.tensor(encoding["input_ids"]).unsqueeze(0).cuda()

        return 0

    def test_step(self, batch, batch_idx):
        vlmo_utils.set_task(self)
        if batch['qid'][0] == 446459000:
            self.target_img = batch['image'][0]
            self.target_txt = batch['text'][0]
        if int(batch['qid'][0] / 100) in [4464590, 1670440]:
            return dict()
        if batch['qid'][0] in self.right_list:
            ret = dict()
            ret['preds'] = [self.vlmo_ans_table[str(batch['qid'][0])]]
            if ret['preds'][0] not in batch['vqa_answer'][
                0]:  # batch['vqa_scores'][0][batch['vqa_answer'][0].index(ret['preds'][0])] == max(batch['vqa_scores'][0]):
                print('not_alogned')
                return dict()
            if batch['vqa_scores'][0][batch['vqa_answer'][0].index(ret['preds'][0])] != max(batch['vqa_scores'][0]):
                print('not aligned')
                return dict()
            ##################################Stat Attack############################
            attack_batch = copy.deepcopy(batch)
            vilt_ans = self.vilt_ans_table[str(int(batch['qid'][0]))]
            all_correct_ans = self.all_correct_ans[str(int(batch['qid'][0]))]
            ans = copy.deepcopy(ret['preds'][0])
            self.batch = copy.deepcopy(attack_batch)
            paraphrase_text = self.chatgpt[str(batch['qid'][0])][1]
            paraphrase_words = paraphrase_text.strip('.').split(' ')
            # print(paraphrase_words)
            old_alg = 1
            ans_words, ans_sub_words, ans_keys = self._tokenize(vilt_ans.lower(), self.tokenizer_mlm)
            attack_ans_words = self.filter(ans_words)
            pa_words, pa_sub_words, pa_keys = self._tokenize(paraphrase_text.strip('.').lower(), self.tokenizer_mlm)
            pa_text = copy.deepcopy(pa_words)
            gt_sentence = ' '.join(pa_text) + '.'
            encoding_gt = self.tokenizer_mlm(
                gt_sentence,
                padding="max_length",
                truncation=True,
                max_length=40,
                return_special_tokens_mask=True,
            )
            encode_ids = torch.tensor(encoding_gt["input_ids"]).unsqueeze(0).cuda()
            batch["text_labels_mlm"] = copy.deepcopy(attack_batch['text_labels'])
            mask_pos_list = []
            mask_word_list = []
            vilt_sub_word_length_lst = []
            vilt_ans_word_lst = []
            # print(paraphrase_words,pa_words,paraphrase_text)
            # exit()
            for ans_split in attack_ans_words:
                if ans_split in pa_words:
                    vilt_ans_word_lst.append(ans_split)
                    old_alg = 0
                    mask_pos = pa_words.index(ans_split)
                    mask_pos_list.append(mask_pos)

                    sub_words_length = pa_keys[mask_pos][-1] - pa_keys[mask_pos][0]
                    vilt_sub_word_length_lst.append(sub_words_length)
                    substitute = ['[MASK]' for i in range(sub_words_length)]
                    mask_word_list.append(substitute)
            vilt_pos_list = copy.deepcopy(mask_pos_list)
            if old_alg == 1:
                mlm_labels = None
            else:
                # if len(all_correct_ans)==0:
                list_words = copy.deepcopy(pa_words)
                sorted_id = sorted(range(len(mask_pos_list)), key=lambda k: mask_pos_list[k], reverse=True)
                mask_pos_list.sort(reverse=True)
                new_mask_word_list = [mask_word_list[id] for id in sorted_id]
                end_pos_ques = 0
                multi_synom = 0
                for mp, sub in zip(mask_pos_list, new_mask_word_list):
                    list_words = list_words[0:mp] + sub + list_words[mp + 1:]
                    batch["text_labels_mlm"][0][pa_keys[mp][0] + 1 + end_pos_ques:pa_keys[mp][1] + 1 + end_pos_ques] = \
                        encode_ids[0][
                        pa_keys[mp][
                            0] + 1:
                        pa_keys[mp][
                            1] + 1]
                batch['text_mlm'] = ' '.join(list_words) + '.'
                encoding = self.tokenizer_mlm(
                    batch['text_mlm'],
                    padding="max_length",
                    truncation=True,
                    max_length=40,
                    return_special_tokens_mask=True,
                )

                attack_batch[f"text_mask_mlm"] =  torch.tensor(encoding["attention_mask"]).unsqueeze(0).cuda()
                attack_batch[f"text_ids_mlm"] = torch.tensor(encoding["input_ids"]).unsqueeze(0).cuda()
                self.batch[f"text_ids_mlm"] = copy.deepcopy(attack_batch["text_ids_mlm"])
                self.batch[f"text_mask_mlm"] = copy.deepcopy(attack_batch["text_mask_mlm"])
                if len(all_correct_ans)==1:
                    mlm_labels = batch['text_labels_mlm']
                    # continue
                    # print(mlm_labels.shape,mlm_labels.device,type(mlm_labels))
                elif len(all_correct_ans)>1:
                    mlm_labels_lst=[]
                    mlm_labels_lst.append(batch['text_labels_mlm'])
                    if ans not in all_correct_ans:
                        print('wrong correct, because the answer is not in all_correct_ans',ans,all_correct_ans)
                    # print('all',all_correct_ans,vilt_ans)
                    for cand_ans in all_correct_ans:
                        # if exit_flag:
                        #     break
                        mlm_flag = True
                        cand_mask_pos_list = []
                        cand_mask_word_list = []
                        cand_ans_words, cand_ans_sub_words, cand_ans_keys = self._tokenize(cand_ans.lower(), self.tokenizer_mlm)
                        cand_attack_ans_words = self.filter(cand_ans_words)
                        if len(cand_attack_ans_words) !=len(vilt_ans_word_lst):
                            # print('stop_here')
                            continue
                        if cand_ans==vilt_ans:
                            continue
                        for i,cand_ans_split in enumerate(cand_attack_ans_words):
                            # if ans_split in paraphrase_words:
                            #     old_alg = 0
                            #     mask_pos = paraphrase_words.index(ans_split)
                            #     mask_pos_list.append(mask_pos)
                            cand_sub_words_length = cand_ans_keys[i][-1] - cand_ans_keys[i][0]
                            # print(cand_sub_words_length,vilt_sub_word_length_lst)
                            if cand_sub_words_length!=vilt_sub_word_length_lst[i]:
                                # print('stop_hhhh')
                                mlm_flag=False
                                break
                            cand_substitute = ['[MASK]' for i in range(cand_sub_words_length)]
                            cand_mask_word_list.append(cand_substitute)
                            cand_mask_pos_list.append(vilt_pos_list[i])
                        if mlm_flag==True:

                            sorted_id = sorted(range(len(cand_mask_pos_list)), key=lambda k: cand_mask_pos_list[k], reverse=True)
                            cand_mask_pos_list.sort(reverse=True)
                            cand_new_mask_word_list = [cand_mask_word_list[id] for id in sorted_id]
                            end_pos_ques = 0
                            multi_synom = 0
                            # cand_mlm_labels=copy.deepcopy()
                            cand_mlm_labels=copy.deepcopy(attack_batch['text_labels'])
                            cand_pa_words, cand_pa_sub_words, cand_pa_keys = self._tokenize(paraphrase_text.strip('.').lower(),
                                                                             self.tokenizer_mlm)
                            cand_list_words = copy.deepcopy(cand_pa_words)
                            for (cli,cwl) in zip(cand_mask_pos_list,cand_attack_ans_words):
                                cand_pa_words[cli]=cwl
                            cand_pa_text = copy.deepcopy(cand_pa_words)
                            cand_gt_sentence = ' '.join(cand_pa_text)  + '.'
                            cand_encoding_gt = self.tokenizer_mlm(
                                cand_gt_sentence,
                                padding="max_length",
                                truncation=True,
                                max_length=40,
                                return_special_tokens_mask=True,
                            )
                            cand_encode_ids = torch.tensor(cand_encoding_gt["input_ids"]).unsqueeze(0).cuda()
                            for mp, sub in zip(cand_mask_pos_list, cand_new_mask_word_list):
                                cand_list_words = cand_list_words[0:mp] + sub + cand_list_words[mp + 1:]
                                cand_mlm_labels[0][
                                cand_pa_keys[mp][0] + 1 + end_pos_ques:cand_pa_keys[mp][1] + 1 + end_pos_ques] = \
                                    cand_encode_ids[0][
                                    cand_pa_keys[mp][
                                        0] + 1:
                                    cand_pa_keys[mp][
                                        1] + 1]
                            mlm_labels_lst.append(cand_mlm_labels)
                    if len(mlm_labels_lst)==1:
                        mlm_labels = mlm_labels_lst[0]
                    else:
                        mlm_labels = torch.stack(mlm_labels_lst,axis=1)
                    # print('mlm_labels',mlm_labels.shape,len(mlm_labels.size()))
                    # continue
            self.predict = copy.deepcopy(ret)
            loss = 0
            tt = copy.deepcopy(batch['image'][0])
            tgt_batch = copy.deepcopy(attack_batch)
            tgt_feats, feats_list, feats_list_img = self.Gen_ori_feats(tgt_batch)
            # exit()
            # print('tttt',tgt_feats.shape,feats_list.shape,feats_list_img.shape)
            # exit()
            adv_img = copy.deepcopy(attack_batch['image'][0])
            tgt_last = None
            feats_last = None
            cos_list = []
            loss_list = []
            # sort_bank=copy.deepcopy(text_bank)
            idx = 1
            ii = 0
            if old_alg == 0:
                logits = self.pgd_mlm_attack(adv_img)[0]
                end_pos = torch.where(attack_batch[f"text_ids_mlm"][0] == 102)[0].detach().cpu().numpy()
                id_sentence = copy.deepcopy(attack_batch["text_ids_mlm"][0][:end_pos[0]])
                m_pos = pa_keys[mask_pos][0] + 1  # +end_pos_ques
                for id in range(sub_words_length):
                    pred_word = torch.argmax(logits[0, m_pos]).detach().cpu()
                    id_sentence[m_pos] = pred_word
                    m_pos += 1
                before_attack = self.tokenizer_mlm.decode(id_sentence[1:])
                # print('before_attack',before_attack)
            count_words = 0
            mask_list = []
            word_word_list = []
            iter_list, bert_cand_list = self.cal_text_attack_list(attack_batch['text'][0])
            ori_words, ori_sub_words, ori_keys = self._tokenize(batch['text'][0].strip('?').lower(),
                                                                self.tokenizer_mlm)
            attack_vector = []
            sub_list = []
            for idx, (ori_key, bert_cand) in enumerate(zip(ori_keys, bert_cand_list)):
                if bert_cand is not None:
                    attack_vector.append(ori_key[0] + 1)
                    sub_list.append(idx)
            encoding = self.tokenizer_mlm(
                attack_batch['text'][0],
                padding="max_length",
                truncation=True,
                max_length=40,
                return_special_tokens_mask=True,
            )
            adv_text = copy.deepcopy(attack_batch['text'][0])
            attack_batch[f"text_ids"] = torch.tensor(encoding["input_ids"]).unsqueeze(0).cuda()
            ori_emb = self.text_embeddings(batch[f"text_ids"])
            ori_text = copy.deepcopy(attack_batch['text'][0])
            attack_batch[f"text_masks"] = torch.tensor(encoding["attention_mask"]).unsqueeze(0).cuda()
            if len(iter_list) == 0:
                if old_alg == 1:
                    torch.set_grad_enabled(True)
                    adv_x, loss = pgd.projected_gradient_descent(self.pgd_attack, adv_img, 0.125, 0.01, 40,
                                                                 np.inf, clip_min=-1, clip_max=1,
                                                                 y=[None, tgt_feats, feats_list, None, None],
                                                                 time=ii, ori_x=batch['image'][0], ls=old_alg)
                    torch.set_grad_enabled(False)
                if old_alg == 0:
                    torch.set_grad_enabled(True)
                    adv_x, loss = pgd.projected_gradient_descent([self.pgd_attack, self.pgd_mlm_attack], adv_img,
                                                                 0.125, 0.01, 20,
                                                                 np.inf, clip_min=-1, clip_max=1,
                                                                 y=[mlm_labels, tgt_feats, feats_list, None],
                                                                 time=ii, ori_x=batch['image'][0], ls=old_alg)
                    torch.set_grad_enabled(False)
            else:
                for iter_idx, iter in enumerate(iter_list):
                    torch.set_grad_enabled(True)
                    if old_alg == 1:
                        adv_encoding = self.tokenizer_mlm(
                            adv_text,
                            padding="max_length",
                            truncation=True,
                            max_length=40,
                            return_special_tokens_mask=True,
                        )
                        self.batch[f"text_ids"] = torch.tensor(adv_encoding["input_ids"]).unsqueeze(0).cuda()
                        self.batch[f"text_masks"] = torch.tensor(adv_encoding["attention_mask"]).unsqueeze(0).cuda()
                        # self.batch = copy.deepcopy(batch)
                        adv_x, loss = pgd.projected_gradient_descent(self.pgd_attack, adv_img, 0.125, 0.01, iter,
                                                                     np.inf, clip_min=-1, clip_max=1,
                                                                     y=[None, tgt_feats, feats_list, None, None],
                                                                     time=ii, ori_x=batch['image'][0], ls=old_alg)
                        if iter_idx == len(iter_list) - 1:
                            torch.set_grad_enabled(False)
                            adv_img = adv_x
                            ii = 1
                            break

                        else:
                            adv_text_ids = torch.tensor(adv_encoding["input_ids"]).unsqueeze(0).cuda()
                            adv_text_embeds = self.text_embeddings(adv_text_ids)
                            adv_x, text_embed_gradient = pgd_vl.projected_gradient_descent(self.pgd_attack_vl,
                                                                                           [adv_x, adv_text_embeds],
                                                                                           0.125, 0.01, 1,
                                                                                           np.inf, clip_min=-1,
                                                                                           clip_max=1,
                                                                                           y=[None, tgt_feats,
                                                                                              feats_list, None, None],
                                                                                           time=1,
                                                                                           ori_x=batch['image'][0],
                                                                                           ls=1,
                                                                                           attack_mask=attack_vector)
                            adv_text, _ = self.update_adv_text(text_embed_gradient, bert_cand_list, sub_list, adv_text,
                                                               attack_vector, ori_emb, ori_text)
                    else:
                        adv_encoding = self.tokenizer_mlm(
                            adv_text,
                            padding="max_length",
                            truncation=True,
                            max_length=40,
                            return_special_tokens_mask=True,
                        )
                        self.batch[f"text_ids"] = torch.tensor(adv_encoding["input_ids"]).unsqueeze(0).cuda()
                        self.batch[f"text_masks"] = torch.tensor(adv_encoding["attention_mask"]).unsqueeze(0).cuda()
                        adv_x, loss = pgd.projected_gradient_descent([self.pgd_attack, self.pgd_mlm_attack], adv_img,
                                                                     0.125, 0.01, int(iter / 2),
                                                                     np.inf, clip_min=-1, clip_max=1,
                                                                     y=[mlm_labels, tgt_feats, feats_list, None],
                                                                     time=ii, ori_x=batch['image'][0], ls=old_alg)
                        if iter_idx == len(iter_list) - 1:
                            torch.set_grad_enabled(False)
                            adv_img = adv_x
                            ii = 1
                            break
                        else:
                            adv_text_ids = torch.tensor(adv_encoding["input_ids"]).unsqueeze(0).cuda()
                            adv_text_embeds = self.text_embeddings(adv_text_ids)
                            adv_x, text_embed_gradient = pgd_vl.projected_gradient_descent(self.pgd_attack_vl,
                                                                                           [adv_x, adv_text_embeds],
                                                                                           0.125, 0.01, 1,
                                                                                           np.inf, clip_min=-1,
                                                                                           clip_max=1,
                                                                                           y=[None, tgt_feats,
                                                                                              feats_list, None, None],
                                                                                           time=1,
                                                                                           ori_x=batch['image'][0],
                                                                                           ls=1,
                                                                                           attack_mask=attack_vector)
                            adv_text, op_mlm_list = self.update_adv_text(text_embed_gradient, bert_cand_list, sub_list,
                                                                         adv_text, attack_vector, ori_emb, ori_text)
                            self.update_mlm_text(op_mlm_list, list_words)
                        end_pos = torch.where(attack_batch[f"text_ids_mlm"][0] == 102)[0].detach().cpu().numpy()
                        id_sentence = copy.deepcopy(attack_batch["text_ids_mlm"][0][:end_pos[0]])

                        word_list_tensor = []
                        m_pos = pa_keys[mask_pos][0] + 1  # +end_pos_ques
                        # print(id_sentence, m_pos)
                        logits = self.pgd_mlm_attack(adv_x)[0]
                        for id in range(sub_words_length):
                            pred_word = torch.argmax(logits[0, m_pos]).detach().cpu()
                            word_list_tensor.append(pred_word)
                            id_sentence[m_pos] = pred_word
                            # print('pred', pred_word)
                            # exit()
                            m_pos += 1
                        attack_sentence = self.tokenizer_mlm.decode(id_sentence[1:])
                        # print('after_attack', attack_sentence)

                    torch.set_grad_enabled(False)
                    adv_img = adv_x
                    ii = 1
                # print(adv_x.shape)

            attack_dict = {'image': adv_x, 'text': adv_text}
            torch.save(adv_x.cpu().detach(),
                       os.path.join(self.adv_store_img_source, str(int(batch['qid'][0])) + '.pt'))
            self.adv_txt_dict[str(int(batch['qid'][0]))] = adv_text
            self.attack_dict[str(batch['qid'][0])] = attack_dict
            if len(self.attack_dict) == 10:
                self.reload_vqa()
                self.to(self.device)
                for qid_key in self.attack_dict.keys():
                    adv_image = self.attack_dict[qid_key]['image']
                    adv_txt = self.attack_dict[qid_key]['text']
                    adv_encoding = self.tokenizer_mlm(
                        adv_txt,
                        padding="max_length",
                        truncation=True,
                        max_length=40,
                        return_special_tokens_mask=True,
                    )
                    self.batch[f"text_ids"] = torch.tensor(adv_encoding["input_ids"]).unsqueeze(0).cuda()
                    self.batch[f"text_masks"] = torch.tensor(adv_encoding["attention_mask"]).unsqueeze(0).cuda()
                    adv_cls_feats = self.pgd_attack(adv_image)[0]
                    logits = self.vqa_classifier(adv_cls_feats)
                    out_v = objectives.vqa_test_step_after_pgd(self, logits)
                    if out_v['preds'][0] != self.vlmo_ans_table[str(qid_key)]:
                        self.acc_list.append(1)
                        if old_alg == 1:
                            self.count_kdd += 1
                    else:
                        self.acc_list.append(0)
                self.reload_pretrain()
                self.to(self.device)
                self.attack_dict = {}
                if len(self.acc_list) % 50 == 0 and len(self.acc_list) != 0:
                    print('attack_accuracy', sum(self.acc_list) / len(self.acc_list))
        ret = dict()
        return ret

    def test_epoch_end(self, outs):
        with open(self.adv_store_txt_source, 'w') as file:
            file.write(json.dumps(self.adv_txt_dict))
        self.reload_vqa()
        self.to(self.device)
        for qid_key in self.attack_dict.keys():
            adv_image = self.attack_dict[qid_key]['image']
            adv_txt = self.attack_dict[qid_key]['text']
            adv_encoding = self.tokenizer_mlm(
                adv_txt,
                padding="max_length",
                truncation=True,
                max_length=40,
                return_special_tokens_mask=True,
            )
            self.batch[f"text_ids"] = torch.tensor(adv_encoding["input_ids"]).unsqueeze(0).cuda()
            self.batch[f"text_masks"] = torch.tensor(adv_encoding["attention_mask"]).unsqueeze(0).cuda()
            adv_cls_feats = self.pgd_attack(adv_image)[0]
            logits = self.vqa_classifier(adv_cls_feats)
            out_v = objectives.vqa_test_step_after_pgd(self, logits)
            if out_v['preds'][0] != self.vlmo_ans_table[str(qid_key)]:
                self.acc_list.append(1)
            else:
                self.acc_list.append(0)
        self.reload_pretrain()
        self.to(self.device)
        self.attack_dict = {}
        if len(self.acc_list) != 0:
            print('acc_vqa', sum(self.acc_list) / len(self.acc_list), len(self.acc_list))
        exit()
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name, self.hparams.config["log_dir"])
        vlmo_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vlmo_utils.set_schedule(self)
