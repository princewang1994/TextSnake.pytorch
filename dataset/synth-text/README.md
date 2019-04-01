## SynthText

SynthText is uploaded to Baidu Cloud [link](链接:https://pan.baidu.com/s/17Gk301SwsnoESM1jQZRq0g), extract code `tb5g`

1. download from link above and unzip it SynthText.zip
2. make traing list by running: `python dataset/synth-text/make_list.py`
3. pretrain using synthtext:

```bash
CUDA_VISIBLE_DEVICES=$GPUID python train.py synthtext_pretrain --dataset synth-text --viz --max_epoch 1 --batch_size 8
```