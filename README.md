# TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes

A PyTorch implement of **TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes** (ECCV 2018) by `Face++`

- Paper link: [arXiv:1807.01544](https://arxiv.org/abs/1807.01544)
- Github: [princewang1994/TextSnake.pytorch](https://github.com/princewang1994/TextSnake.pytorch)
- Blog: [TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes](http://blog.prince2015.club/2019/01/06/TextSnake/)

<div style="color:#0000FF" align="center">
<img src="http://princepicbed.oss-cn-beijing.aliyuncs.com/blog_20190120144708.png" width="630"/> 
</div>

## Paper


![](http://princepicbed.oss-cn-beijing.aliyuncs.com/blog_20181228172334.png)

Comparison of diﬀerent representations for text instances. (a) Axis-aligned rectangle. (b) Rotated rectangle. (c) Quadrangle. (d) TextSnake. Obviously, the proposed TextSnake representation is able to eﬀectively and precisely describe the geometric properties, such as location, scale, and bending of curved text with perspective distortion, while the other representations (axis-aligned rectangle, rotated rectangle or quadrangle) struggle with giving accurate predictions in such cases.

<div style="color:#0000FF" align="center">
<img src="http://princepicbed.oss-cn-beijing.aliyuncs.com/blog_20181228172346.png" width="530"/> 
</div>


Text snake element:

- center point
- tangent line
- text region

## Description

Generally, this code has following features:

1. include complete training and inference code
2. pure python version without extra compiling
3. compatible with laste PyTorch version (write with pytroch 0.4.0)
4. support [TotalText](https://github.com/cs-chan/Total-Text-Dataset) dataset


## Getting Started

This repo includes the training code and inference demo of TextSnake, training and infercence can be simplely run with a  few code. 

### Prerequisites

To run this repo successfully, it is highly recommanded with:

- **Linux (Ubuntu 16.04)**
- **Python3.6**
- **Anaconda3**
- **NVIDIA GPU(with 8G or larger GPU memory for training, 2G for inference)**

(I haven't test it on other Python version.)

1. clone this repository

```shell
git clone https://github.com/princewang1994/TextSnake.pytorch.git
```

2. python package can be installed with `pip`
```shell
$ cd $TEXTSNAKE
$ pip install -r requirements.txt
```

## Data preparation

* `Total-Text`: follow the [dataset/total_text/README.md](dataset/total_text/README.md)
* `SynthText`: follow the [datset/synth-text/README.md](dataset/synth-text/README.md)

## Pretraining with SynthText

```shell
$ CUDA_VISIBLE_DEVICES=$GPUID python train.py synthtext_pretrain --dataset synth-text --viz --max_epoch 1 --batch_size 8
```

## Training

Training model with given experiment name `$EXPNAME`

training from scratch:
```shell
$ EXPNAME=example
$ CUDA_VISIBLE_DEVICES=$GPUID python train.py $EXPNAME --viz
```

training with pretrained model(improved performance much)
```shell
$ EXPNAME=example
$ CUDA_VISIBLE_DEVICES=$GPUID python train.py example --viz --batch_size 8 --resume save/synthtext/textsnake_vgg_0.pth
```

**options:**

- `exp_name`: experiment name, used to identify different training process
- `--viz`: visualization toggle, output pictures are saved to './vis' by default

other options can be show by run `python train.py -h`

## Running the tests

Runing following command can generate demo on TotalText dataset (300 pictures), the result are save to `./vis` by default

```shell
$ EXPNAME=example
$ CUDA_VISIBLE_DEVICES=$GPUID python demo.py $EXPNAME --checkepoch 190
```

**options:**

- `exp_name`: experiment name, used to identify different training process

other options can be show by run `python train.py -h`

## Evaluation

Total-Text metric is included in `dataset/total_text/Evaluation_Protocol/Python_scripts/Deteval.py`, you should first modify the `input_dir` in `Deteval.py` and run following command for computing DetEval:

```shell
$ python dataset/total_text/Evaluation_Protocol/Python_scripts/Deteval.py
```

which will output 

## Performance

- left: prediction/ground true
- middle: text region(TR)
- right: text center line(TCL)

![](demo/img93.jpg)

![](demo/img96.jpg)

![](demo/img191.jpg)

![](demo/img600.jpg)

## What is comming

- [x] Pretraining with SynthText
- [x] Metric computing
- [ ] Pretrained model upload (soon)
- [ ] More dataset suport: [ICDAR15]

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgement

This project is writen by [Prince Wang](https://github.com/princewang1994), part of codes refer to [songdejia/EAST](https://github.com/songdejia/EAST)