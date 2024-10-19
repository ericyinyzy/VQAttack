1. Follw the [VLMO](https://github.com/microsoft/unilm/tree/master/vlmo)
and [cleverhans](https://github.com/cleverhans-lab/cleverhans)
to set the environment.

2. Set the dataset path `data_root` in vlmo/config.py. We follow `VLMO` to adopt `VQA_Arrow` data type.

3. set the pretrained model path `pretrain_path`, fine-tuned vqa model path `load_path` and in the adv_attack.py,
Our codes are implemented in the `vlmo/vlmo_module.py`, and we update image perturbations
in the '/cleverhans/' dir.

4. To perform VQAttack, run:

```
python run.py with task_finetune_vqa_base_image480 test_only=True
```


5. After runing the attack, the adversarial image embeddings are stored in `attack_dir_VLMO_BASE`.
 The adversarial text is stored in `adv_txt_dict_VLMO_BASE.txt`.

