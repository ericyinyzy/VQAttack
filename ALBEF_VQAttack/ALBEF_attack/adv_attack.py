
import math
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sys
import os
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertForMaskedLM, BertTokenizer
from transformers.models.bert.modeling_bert import BertConfig#, BertEmbeddings
config_atk = BertConfig.from_pretrained('bert-base-uncased')
from models.xbert import BertConfig,BertEmbeddings
from typing import Dict, Iterable, Optional
import copy
import torch
import torch.nn
import utils
import torch.optim
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from filter_words import filter_words
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
filter_words = filter_words + stopwords.words('english')+['?','.']
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
import json

sys.path.append('../cleverhans')
import cleverhans.torch.attacks.projected_gradient_descent as pgd
import cleverhans.torch.attacks.projected_gradient_descent_vl as pgd_vl

class Adv_attack:
    def __init__(self, vqa_model,pretrain_model,tokenizer,device,config):
        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.attack_dict = {}
        self.acc_list=[]
        self.tokenizer = tokenizer
        self.tokenizer_mlm = BertTokenizer.from_pretrained("bert-base-uncased",
                                                           do_lower_case="uncased" in "bert-base-uncased")
        f = open('right_part.txt', 'r')
        a = list(f)
        f.close()
        f = open('right_part_after.txt', 'r')
        a1 = list(f)
        f.close()
        self.adv_store_img_source = 'attack_dir' #store adversarial image
        self.adv_store_txt_source = 'adv_txt_dict_albef.txt' #store adversarial text
        self.adv_txt_dict = {}
        if not os.path.exists(self.adv_store_img_source):
            os.makedirs(self.adv_store_img_source)
        self.correct_list = [int(l.strip('\n')) for l in a] + [int(l.strip('\n')) for l in a1]
        with open('vilt_ans_table_for_chatgpt.txt', 'r') as f:
            self.vilt_ans_table = json.load(f)
        with open('vilt_ans_table_for_chatgpt_after.txt', 'r') as f:
            self.vilt_ans_table.update(json.load(f))
        with open('albef_ans_table.txt', 'r') as f:
            self.tcl_ans_table = json.load(f)
        with open('albef_ans_table_after.txt', 'r') as f:
            self.tcl_ans_table.update(json.load(f))
        with open('chatgpt_all_5k.txt', 'r') as f:
            self.chatgpt = json.load(f)
        with open('chatgpt_all_5k_after.txt', 'r') as f:
            self.chatgpt.update(json.load(f))
        with open('all_correct_ans.txt', 'r') as f:
            self.all_correct_ans = json.load(f)
        with open('all_correct_ans_after.txt', 'r') as f:
            self.all_correct_ans.update(json.load(f))
        self.white_model=copy.deepcopy(pretrain_model)

        checkpoint = torch.load("pretrain model path", map_location="cpu")
        state_dict = checkpoint['model']
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], self.white_model.visual_encoder)
        m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                         self.white_model.visual_encoder_m)

        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped

        self.white_model.load_state_dict(state_dict)
        self.embeddings=self.white_model.text_encoder.bert.embeddings
        self.embeddings_cpu=copy.deepcopy(self.embeddings).to('cpu')
        self.black_model=copy.deepcopy(vqa_model)
        checkpoint = torch.load("fine-tune model path", map_location="cpu")
        state_dict = checkpoint['model']
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], vqa_model.visual_encoder)
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        msg = self.black_model.load_state_dict(state_dict, strict=False)
        with tf.device('cpu'):
            self.USE_model = hub.load(
                '/tfhub_pretrained/universal-sentence-encoder-large_5')
        self.device=device
        self.batch=None
        self.captions=None
        self.vqa_score=0
        self.acc_list=[]

        self.mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config_atk).to(self.device)
    def Gen_ori_feats(self, batch):
        image=batch['image'].to(self.device, non_blocking=True)
        question_input = self.tokenizer(batch['question'], padding='longest', truncation = True,max_length = 25,return_tensors="pt",).to(self.device)
        img_feats_list,txt_feats_list = self.white_model.Gen_feats(image,question_input.input_ids,question_input.attention_mask)
        img_feats=torch.cat(img_feats_list, axis=0)
        txt_feats=torch.cat(txt_feats_list, axis=0)

        return img_feats,txt_feats
    def pgd_attack(self,x):
        text_ids = self.batch[f"text_ids"]
        text_masks = self.batch[f"text_masks"]
        img_feats_list, txt_feats_list = self.white_model.Gen_feats(x, text_ids,
                                                                    text_masks)
        img_feats = torch.cat(img_feats_list, axis=0)
        txt_feats = torch.cat(txt_feats_list, axis=0)
        return [txt_feats,img_feats]
    def cal_vqa(self,ans,ans_set,ans_set_score):
        if ans[0] in ans_set:
            self.vqa_score+=ans_set_score[ans_set.index(ans[0])]
    def pgd_mlm_attack(self, x):
        image_token_type_idx = 1
        mask_image = False
        image_embeds = None
        image_masks = None
        mask_text = True
        do_mlm = "_mlm" if mask_text else ""
        text_ids = self.batch[f"text_ids{do_mlm}"]
        text_masks = self.batch[f"text_mask_mlm"]
        mlm_logits=self.white_model.get_mlm_logits(x,text_ids,text_masks)
        return [mlm_logits]
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
    def filter(self, ori_words):
        stop_words = ['on', 'and', 'in', 'his', 'her', 'its']
        for i in stop_words:
            if i in ori_words:
                ori_words.remove(i)
        return ori_words
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
    def pgd_attack_vl(self, x):
        text_ids = self.batch[f"text_ids"]
        text_masks = self.batch[f"text_masks"]
        img_feats_list, txt_feats_list = self.white_model.Gen_feats_from_embeds(x[0], x[1],text_ids,text_masks)
        img_feats = torch.cat(img_feats_list, axis=0)
        txt_feats = torch.cat(txt_feats_list, axis=0)
        return [txt_feats,img_feats]
    def cal_text_attack_list(self, ori_text):
        iter_list = []
        bert_cand_list = []
        text = ori_text.lower()
        feature = Feature(text)
        tokenizer = self.tokenizer_mlm
        # words, sub_words, keys = self._tokenize(feature.seq.strip('.'), tokenizer)
        words, sub_words, keys = self._tokenize(ori_text.lower(), self.tokenizer_mlm)
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
    def update_adv_text(self, text_embed_gradient, bert_cand_list, cand_wd_list, adv_text, attack_vector, ori_emb,
                        ori_text):
        words, sub_words, keys = self._tokenize(adv_text.lower(), self.tokenizer_mlm)
        ori_words = copy.deepcopy(words)
        adv_words = copy.deepcopy(words)
        sort_list = []
        dir_sim_list = []
        occupied_list = []
        # print('words',adv_text,words,cand_wd_list)
        for idx, (cand_wd_idx, sub_wd_idx) in enumerate(zip(cand_wd_list, attack_vector)):
            attack_grad = text_embed_gradient[0, idx]
            cand_list = bert_cand_list[cand_wd_idx]
            cand_words_words = copy.deepcopy(words)
            for idd, cand_wd in enumerate(cand_list):
                sort_list.append([cand_wd_idx, idd])
                if cand_wd_idx >= len(cand_words_words):
                    print('onebug', adv_text.lower(), words, ori_text, cand_wd_list, bert_cand_list,
                          attack_vector)
                    return self.tokenizer_mlm.convert_tokens_to_string(ori_words), []
                cand_words_words[cand_wd_idx] = cand_wd
                adv_sentence = ' '.join(cand_words_words) #+ '?'
                encoding_adv = self.tokenizer_mlm(
                    adv_sentence,
                    padding='longest',
                    truncation=True,
                    max_length=25,
                    return_tensors="pt"
                )
                # print(encoding_adv["input_ids"])
                encode_ids_adv = encoding_adv["input_ids"].cuda()
                adv_text_embeds = self.text_embeddings(encode_ids_adv)
                cand_emb_dir = adv_text_embeds[0, sub_wd_idx] - ori_emb[0, sub_wd_idx]
                dir_sim = self.dir_sim(cand_emb_dir, attack_grad)
                dir_sim_list.append(dir_sim)
                # print('words',cand_words_words)
        ll = sorted(range(len(dir_sim_list)), key=lambda k: dir_sim_list[k], reverse=True)
        sorted_op_list = [sort_list[i] for i in ll]
        sorted_sim_list = [dir_sim_list[i] for i in ll]
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
            temp_text = self.tokenizer_mlm.convert_tokens_to_string(temp_replace) #+ '?'
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
        return self.tokenizer_mlm.convert_tokens_to_string(adv_words), op_mlm_list
    def dir_sim(self, cand_emb_dir, attack_grad):
        # print(cand_emb_dir.shape,attack_grad.shape)

        cand_norm = F.normalize(cand_emb_dir, p=2, dim=0)
        attack_norm = F.normalize(attack_grad, p=2, dim=0)
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        dir_sim = cos(cand_norm, attack_norm)

        return dir_sim
    def update_mlm_text(self, op_mlm_list, list_words):
        for op_mlm in op_mlm_list:
            ori_word = op_mlm[0]
            cand_word = op_mlm[1]
            if ori_word in list_words:
                index_list = [index for (index, value) in enumerate(list_words) if value == ori_word]
                for idx in index_list:
                    list_words[idx] = cand_word
        text_mlm = ' '.join(list_words) #+ '.'
        encoding = self.tokenizer_mlm(
            text_mlm,
            padding='longest',
            truncation=True,
            max_length=25,
            return_tensors="pt"
        )
        self.batch[f"text_mask_mlm"] = encoding["attention_mask"].cuda()
        self.batch[f"text_ids_mlm"] = encoding["input_ids"].cuda()

        return 0
    def text_embeddings_cpu(self,text_input_ids):
        position_ids=None
        token_type_ids = torch.zeros(text_input_ids.size(), dtype=torch.long, device='cpu')
        inputs_embeds=None
        past_key_values_length=0
        embedding_output = self.embeddings_cpu(
            input_ids=text_input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )


        return embedding_output
    def text_embeddings(self,text_input_ids):
        position_ids=None
        token_type_ids = torch.zeros(text_input_ids.size(), dtype=torch.long, device=self.device)
        inputs_embeds=None
        past_key_values_length=0
        # print(text_input_ids.device,token_type_ids.device)
        embedding_output = self.embeddings(
            input_ids=text_input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )


        return embedding_output
    @torch.no_grad()
    def evaluate(
        self,
        # criterion: Optional[torch.nn.Module],
        # postprocessors: Dict[str, torch.nn.Module],
        # weight_dict: Dict[str, float],
        data_loader,
        tokenizer
        # evaluator_list,
        # args,
    ):
        answer_list = [answer + '[SEP]' for answer in data_loader.dataset.answer_list]
        answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(self.device)
        self.tokeizer=tokenizer
        self.white_model.eval()
        self.black_model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Test:"


        # correct_list=[]
        idx=0
        success_count=0
        vqa_score=0
        count_sample=0
        multi_num = 0
        iter_step = 0
        iter_dict = {}
        import json
        print_freq=50000
        for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            if int(batch['question_id'][0]) not in self.correct_list:
                continue
            ret = dict()
            cali_ans=[i[0] for i in batch['answer']]
            cali_weight=[i[0].cpu().numpy() for i in batch['weight']]
            ret['preds'] = [self.tcl_ans_table[str(batch['question_id'][0])]]
            if ret['preds'][0] not in cali_ans:
                print('not_alogned')
                return dict()
            if cali_weight[cali_ans.index(ret['preds'][0])] != max(cali_weight):
                print('not aligned')
                return dict()
            attack_batch = copy.deepcopy(batch)
            vilt_ans = self.vilt_ans_table[str(int(batch['question_id'][0]))]
            all_correct_ans = self.all_correct_ans[str(int(batch['question_id'][0]))]
            ans = copy.deepcopy(ret['preds'][0])
            self.batch = copy.deepcopy(attack_batch)
            paraphrase_text = self.chatgpt[str(int(batch['question_id'][0]))][1]
            paraphrase_words = paraphrase_text.strip('.').split(' ')
            old_alg = 1
            ans_words, ans_sub_words, ans_keys = self._tokenize(vilt_ans.lower(), self.tokenizer_mlm)
            attack_ans_words = self.filter(ans_words)
            pa_words, pa_sub_words, pa_keys = self._tokenize(paraphrase_text.strip('.').lower(), self.tokenizer_mlm)
            pa_text = copy.deepcopy(pa_words)
            gt_sentence = ' '.join(pa_text) #+ '.'
            encoding_gt = self.tokenizer_mlm(
                gt_sentence,
                padding='longest',
                truncation=True,
                max_length=25,
                return_tensors="pt"
            )
            encode_ids = encoding_gt["input_ids"].cuda()
            attack_batch['text_labels']=torch.full_like(encode_ids,-100)
            batch["text_labels_mlm"] = copy.deepcopy(attack_batch['text_labels'])
            mask_pos_list = []
            mask_word_list = []
            vilt_sub_word_length_lst = []
            vilt_ans_word_lst = []
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
                batch['text_mlm'] = ' '.join(list_words)
                encoding = self.tokenizer_mlm(
                    batch['text_mlm'],
                    padding='longest',
                    truncation=True,
                    max_length=25,
                    return_tensors="pt"
                )
                attack_batch[f"text_mask_mlm"] = encoding["attention_mask"].cuda()
                attack_batch[f"text_ids_mlm"] = encoding["input_ids"].cuda()
                self.batch[f"text_ids_mlm"] = copy.deepcopy(attack_batch["text_ids_mlm"])
                self.batch[f"text_mask_mlm"] = copy.deepcopy(attack_batch["text_mask_mlm"])
                if len(all_correct_ans)==1:
                    mlm_labels = batch['text_labels_mlm']
                elif len(all_correct_ans)>1:
                    mlm_labels_lst=[]
                    mlm_labels_lst.append(batch['text_labels_mlm'])
                    if ans not in all_correct_ans:
                        print('wrong correct, because the answer is not in all_correct_ans',ans,all_correct_ans)
                    for cand_ans in all_correct_ans:
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
                            cand_sub_words_length = cand_ans_keys[i][-1] - cand_ans_keys[i][0]
                            if cand_sub_words_length!=vilt_sub_word_length_lst[i]:
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
                            cand_mlm_labels=copy.deepcopy(attack_batch['text_labels'])
                            cand_pa_words, cand_pa_sub_words, cand_pa_keys = self._tokenize(paraphrase_text.strip('.').lower(),
                                                                             self.tokenizer_mlm)
                            cand_list_words = copy.deepcopy(cand_pa_words)
                            for (cli,cwl) in zip(cand_mask_pos_list,cand_attack_ans_words):
                                cand_pa_words[cli]=cwl
                            cand_pa_text = copy.deepcopy(cand_pa_words)
                            cand_gt_sentence = ' '.join(cand_pa_text)  # + '.'
                            cand_encoding_gt = self.tokenizer_mlm(
                                cand_gt_sentence,
                                padding='longest',
                                truncation=True,
                                max_length=25,
                                return_tensors="pt"
                            )
                            cand_encode_ids = cand_encoding_gt["input_ids"].cuda()
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
            self.predict = copy.deepcopy(ret)
            loss = 0
            tt = copy.deepcopy(batch['image'][0])
            tgt_batch = copy.deepcopy(attack_batch)
            ori_img_feats,ori_txt_feats = self.Gen_ori_feats(tgt_batch)
            adv_img = copy.deepcopy(attack_batch['image']).to(self.device, non_blocking=True)
            # image = batch['image'].to(self.device, non_blocking=True)
            tgt_last = None
            feats_last = None
            cos_list = []
            loss_list = []
            idx = 1
            ii = 0
            iter_list, bert_cand_list = self.cal_text_attack_list(attack_batch['question'][0])
            ori_words, ori_sub_words, ori_keys = self._tokenize(batch['question'][0].lower(),
                                                                self.tokenizer_mlm)
            attack_vector = []
            sub_list = []
            for idx, (ori_key, bert_cand) in enumerate(zip(ori_keys, bert_cand_list)):
                if bert_cand is not None:
                    attack_vector.append(ori_key[0] + 1)
                    sub_list.append(idx)
            encoding = self.tokenizer_mlm(
                attack_batch['question'][0],
                padding='longest',
                truncation=True,
                max_length=25,
                return_tensors="pt"
            )
            adv_text = copy.deepcopy(attack_batch['question'][0])
            attack_batch[f"text_ids"] = encoding["input_ids"].cuda()
            self.batch[f"text_ids"] = encoding["input_ids"].cuda()
            encoding_ori = self.tokenizer_mlm(
                batch['question'][0],
                padding='longest',
                truncation=True,
                max_length=25,
                return_tensors="pt"
            )
            batch[f"text_ids"] = encoding_ori["input_ids"].cuda()
            ori_emb = self.text_embeddings(batch[f"text_ids"])
            ori_text = copy.deepcopy(batch['question'][0])
            attack_batch[f"text_masks"] = encoding["attention_mask"].cuda()
            self.batch[f"text_masks"] = encoding["attention_mask"].cuda()

            if len(iter_list) == 0:
                if old_alg == 1:
                    torch.set_grad_enabled(True)
                    adv_x, loss = pgd.projected_gradient_descent(self.pgd_attack, adv_img, 0.125, 0.01, 40,
                                                                 np.inf, clip_min=-1, clip_max=1,
                                                                 y=[ori_txt_feats, ori_img_feats, None, None, None],
                                                                 time=ii, ori_x=batch['image'].cuda(), ls=old_alg)
                    torch.set_grad_enabled(False)
                if old_alg == 0:
                    torch.set_grad_enabled(True)
                    adv_x, loss = pgd.projected_gradient_descent([self.pgd_attack, self.pgd_mlm_attack], adv_img,
                                                                 0.125, 0.01, 20,
                                                                 np.inf, clip_min=-1, clip_max=1,
                                                                 y=[mlm_labels,ori_txt_feats,ori_img_feats],
                                                                 time=ii, ori_x=batch['image'].cuda(), ls=old_alg)
                    torch.set_grad_enabled(False)
            else:
                for iter_idx, iter in enumerate(iter_list):
                    torch.set_grad_enabled(True)
                    if old_alg == 1:
                        adv_encoding = self.tokenizer_mlm(
                            adv_text,
                            padding='longest',
                            truncation=True,
                            max_length=25,
                            return_tensors="pt"
                        )
                        self.batch[f"text_ids"] = adv_encoding["input_ids"].cuda()
                        self.batch[f"text_masks"] = adv_encoding["attention_mask"].cuda()
                        adv_x, loss = pgd.projected_gradient_descent(self.pgd_attack, adv_img, 0.125, 0.01, iter,
                                                                     np.inf, clip_min=-1, clip_max=1,
                                                                     y=[ori_txt_feats, ori_img_feats, None, None, None],
                                                                     time=ii, ori_x=batch['image'].cuda(), ls=old_alg)
                        if iter_idx == len(iter_list) - 1:
                            torch.set_grad_enabled(False)
                            adv_img = adv_x
                            ii = 1
                            break

                        else:
                            adv_text_ids = adv_encoding["input_ids"].cuda()
                            adv_text_embeds = self.text_embeddings(adv_text_ids)

                            adv_x, text_embed_gradient = pgd_vl.projected_gradient_descent(self.pgd_attack_vl,
                                                                                           [adv_x, adv_text_embeds],
                                                                                           0.125, 0.01, 1,
                                                                                           np.inf, clip_min=-1,
                                                                                           clip_max=1,
                                                                                           y=[ori_txt_feats, ori_img_feats,
                                                                                              None, None, None],
                                                                                           time=1,
                                                                                           ori_x=batch['image'].cuda(),
                                                                                           ls=1,
                                                                                           attack_mask=attack_vector)
                            adv_text, _ = self.update_adv_text(text_embed_gradient, bert_cand_list, sub_list, adv_text,
                                                               attack_vector, ori_emb, ori_text)
                    else:
                        adv_encoding = self.tokenizer_mlm(
                            adv_text,
                            padding='longest',
                            truncation=True,
                            max_length=25,
                            return_tensors="pt"
                        )
                        self.batch[f"text_ids"] = adv_encoding["input_ids"].cuda()
                        self.batch[f"text_masks"] = adv_encoding["attention_mask"].cuda()
                        adv_x, loss = pgd.projected_gradient_descent([self.pgd_attack, self.pgd_mlm_attack], adv_img,
                                                                     0.125, 0.01, int(iter / 2),
                                                                     np.inf, clip_min=-1, clip_max=1,
                                                                     y=[mlm_labels,ori_txt_feats,ori_img_feats],
                                                                     time=ii, ori_x=batch['image'].cuda(), ls=old_alg)
                        if iter_idx == len(iter_list) - 1:
                            torch.set_grad_enabled(False)
                            adv_img = adv_x
                            ii = 1
                            break
                        else:
                            adv_text_ids = adv_encoding["input_ids"].cuda()
                            adv_text_embeds = self.text_embeddings(adv_text_ids)
                            adv_x, text_embed_gradient = pgd_vl.projected_gradient_descent(self.pgd_attack_vl,
                                                                                           [adv_x, adv_text_embeds],
                                                                                           0.125, 0.01, 1,
                                                                                           np.inf, clip_min=-1,
                                                                                           clip_max=1,
                                                                                           y=[ori_txt_feats, ori_img_feats,
                                                                                              None, None, None],
                                                                                           time=1,
                                                                                           ori_x=batch['image'].cuda(),
                                                                                           ls=1,
                                                                                           attack_mask=attack_vector)
                            adv_text, op_mlm_list = self.update_adv_text(text_embed_gradient, bert_cand_list, sub_list,
                                                                         adv_text, attack_vector, ori_emb, ori_text)
                            aaa,_,_=self._tokenize(adv_text,self.tokenizer_mlm)
                            self.update_mlm_text(op_mlm_list, list_words)
                        id_sentence = copy.deepcopy(attack_batch["text_ids_mlm"][0])#[:end_pos[0]])

                        word_list_tensor = []
                        m_pos = pa_keys[mask_pos][0] + 1
                        logits = self.pgd_mlm_attack(adv_x)[0]
                        for id in range(sub_words_length):
                            pred_word = torch.argmax(logits[0, m_pos]).detach().cpu()
                            word_list_tensor.append(pred_word)
                            id_sentence[m_pos] = pred_word
                            m_pos += 1
                        attack_sentence = self.tokenizer_mlm.decode(id_sentence[1:])

                    torch.set_grad_enabled(False)
                    adv_img = adv_x
                    ii = 1
            attack_dict = {'image': adv_x, 'text': adv_text}
            torch.save(adv_x.cpu().detach(),os.path.join(self.adv_store_img_source, str(int(batch['question_id'][0])) + '.pt'))
            self.adv_txt_dict[str(int(batch['question_id'][0]))] = adv_text
            self.attack_dict[str(batch['question_id'][0])] = attack_dict
            if len(self.attack_dict) == 10:
                for qid_key in self.attack_dict.keys():
                    adv_image = self.attack_dict[qid_key]['image']
                    adv_txt = self.attack_dict[qid_key]['text']
                    image = adv_image.to(self.device, non_blocking=True)
                    question_input = tokenizer(adv_txt, padding='longest', return_tensors="pt").to(self.device)
                    topk_ids, topk_probs = self.black_model(image, question_input, answer_input, train=False, k=128)
                    for ques_id, topk_id, topk_prob in zip([qid_key], topk_ids, topk_probs):
                        _, pred = topk_prob.max(dim=0)
                        ans_after_attack=data_loader.dataset.answer_list[topk_id[pred]]
                    if ans_after_attack != self.tcl_ans_table[str(qid_key)]:
                        self.acc_list.append(1)
                    else:
                        self.acc_list.append(0)
                self.attack_dict = {}
                if len(self.acc_list) % 50 == 0 and len(self.acc_list) != 0:
                    print('attack_accuracy', sum(self.acc_list) / len(self.acc_list))
        with open(self.adv_store_txt_source, 'w') as file:
            file.write(json.dumps(self.adv_txt_dict))

