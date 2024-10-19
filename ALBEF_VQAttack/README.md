1. Follw the [ALBEF](https://github.com/salesforce/ALBEF)
and [cleverhans](https://github.com/cleverhans-lab/cleverhans)
to set the environment.

2. Set the dataset path in `/ALBEF_demo/configs/VQA.yaml`.
We use the vqa_val.json as the test file.

3. set the pretrained model path in the adv_attack.py,
if you want to perform a transfer attack with same structures,
you need also set the fine-tuned model path.
Our codes are implemented in the adv_attack,py, and we update image perturbations
in the '/cleverhans/' dir.

4. in 'ALBEF_attack' path, run:
```
python VQA.py --config ./configs/VQA.yaml
```

5. After runing the attack, the adversarial image embeddings are stored in `ALBEF/demo/attack_dir`.
 The adversarial text is stored in `adv_txt_dict_albef.txt`.

