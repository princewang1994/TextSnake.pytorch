# TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes

A PyTorch implement of **TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes** (ECCV 2018)

- Paper link: [arXiv:1807.01544](https://arxiv.org/abs/1807.01544)

- Github: [princewang1994/TextSnake.pytorch]()

![](http://princepicbed.oss-cn-beijing.aliyuncs.com/blog_20181228194823.png)

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


## Getting Started

This repo includes the training code and inference demo of TextSnake, training and infercence can be simplely run with a  few code. 

### Prerequisites

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

```shell
EXPNAME=example
CUDA_VISIBLE_DEVICES=$GPUID python train.py $EXPNAME --viz
```

## Running the tests

```shell
EXPNAME=example
CUDA_VISIBLE_DEVICES=$GPUID python demo.py --checkepoch 190
```

## Performance

left: prediction, middle: text region(TR), right: text center line(TCL)

![](demo/24_img650.jpg)

![](demo/26_img612.jpg)

![](demo/13_img637.jpg)

![](demo/107_img600.jpg)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgement

This project is writen by [Prince Wang](https://github.com/princewang1994), some code  refer to [songdejia/EAST](https://github.com/songdejia/EAST)