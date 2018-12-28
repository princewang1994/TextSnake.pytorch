# TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes

A PyTorch implement of **TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes** (Face++ ECCV 2018)

- Paper link: [arXiv:1807.01544](https://arxiv.org/abs/1807.01544)

- Github: [princewang1994/TextSnake.pytorch](https://github.com/princewang1994/TextSnake.pytorch)

![](http://princepicbed.oss-cn-beijing.aliyuncs.com/blog_20181228201251.png)

## Paper

![](http://princepicbed.oss-cn-beijing.aliyuncs.com/blog_20181228172334.png)

Comparison of diﬀerent representations for text instances. (a) Axis-aligned rectangle. (b) Rotated rectangle. (c) Quadrangle. (d) TextSnake. Obviously, the proposed TextSnake representation is able to eﬀectively and precisely describe the geometric properties, such as location, scale, and bending of curved text with perspective distortion, while the other representations (axis-aligned rectangle, rotated rectangle or quadrangle) struggle with giving accurate predictions in such cases.

![](http://princepicbed.oss-cn-beijing.aliyuncs.com/blog_20181228172346.png)

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

- Linux (Ubuntu 16.04)
- Python3.6
- Anaconda3
- NVIDIA GPU(with 8G or larger GPU memroy)

(I haven't test it on other Python version.)

1. clone this repository

```
git clone https://github.com/princewang1994/TextSnake.pytorch.git
```

2. python package can be installed with `pip`
```
cd $TEXTSNAKE
pip install -r requirements.txt
```

## Training

Training model with given experiment name `$EXPNAME`

```shell
EXPNAME=example
CUDA_VISIBLE_DEVICES=$GPUID python train.py $EXPNAME --viz
```

**options:**

- `exp_name`: experiment name, used to identify different training process
- `--viz`: visualization toggle, output pictures are saved to './vis' by default

other options can be show by run `python train.py -h`

## Running the tests

Runing following command can generate demo on TotalText dataset (300 pictures), the result are save to `./vis` by default

```shell
EXPNAME=example
CUDA_VISIBLE_DEVICES=$GPUID python demo.py --checkepoch 190
```

**options:**

- `exp_name`: experiment name, used to identify different training process

other options can be show by run `python train.py -h`

## Performance

left: prediction, middle: text region(TR), right: text center line(TCL)

![](demo/24_img650.jpg)

![](demo/26_img612.jpg)

![](demo/13_img637.jpg)

![](demo/107_img600.jpg)

## What is comming

- [ ] more dataset suport: ICDAR15/[SynthText](https://github.com/ankush-me/SynthText)
- [ ] Metric computing
- [ ] Cython/C++ accelerate core functions

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgement

This project is writen by [Prince Wang](https://github.com/princewang1994), some code  refer to [songdejia/EAST](https://github.com/songdejia/EAST)