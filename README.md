# DeepStroke: Understanding Glyph Structure with Semantic Segmentation and Tabu Search

This work are based on DeepLab V3+.
**Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation**<br />
Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam. arXiv: 1802.02611.<br />
[[DeepLab V3+]](https://arxiv.org/abs/1802.02611). arXiv: 1802.02611, 2018.

## Environment
  PYTHON 2.7

  TensorFlow 1.6.0

  MATLAB 2016b

## Download
Our dataset. [[CCSSD]](http://www.flexifont.com/DeepStroke/CCSSD.zip)

Best checkpoint [[DeepStroke]](http://www.flexifont.com/DeepStroke/DeepStroke.zip). If you want to test on this checkpoint, please unzip it and put the four files in the directory "CCSSD/DATA_GB6763_MULTY_ORDER/MULTY_ORDER2017/exp/train_on_trainval_set/train/"

## How to use
First cd to deeplab

You need replace the directory "/home/wwg/data/CCSSD/" to your real directory to the CCSSD dataset in file train_on_multy_data.sh, vis_on_local.sh and our MATLAB script.

Train model

    sh ./train_on_multy_data.sh 3 4 1,2,3,4

  fist number 3 is used to choose server

  second number 4 is used to set num_clones, it means we use 4 GPU to train
  third parameter 1,2,3,4 means use GPU 1,2,3,4 in the sever

  We will get checkpoint file in the directory "CCSSD/DATA_GB6763_MULTY_ORDER/MULTY_ORDER2017/exp/train_on_trainval_set/train/"

Visualize model

    sh ./vis_on_local.sh

  We will get result in the directory "CCSSD/DATA_GB6763_HLJ/HLJ2017/exp/train_on_trainval_set/vis/", "CCSSD/DATA_GB6763_SS/SS2017/exp/train_on_trainval_set/vis/" and "CCSSD/DATA_GB6763_FZLBJW/FZLBJW2017/exp/train_on_trainval_set/vis/"

Evaluate the model after Visualize model

  cd to /MATLAB/my_script

  use matlab run eval_result_order.m
  

If you find the code and data useful for your research, please consider citing our work:

```
@inproceedings{wang2020deepstroke,
  title={DeepStroke: Understanding Glyph Structure with Semantic Segmentation and Tabu Search},
  author={Wang, Wenguang and Lian, Zhouhui and Tang, Yingmin and Xiao, Jianguo},
  booktitle={International Conference on Multimedia Modeling},
  pages={353--364},
  year={2020},
  organization={Springer}
}
```

## Copyright
The code and dataset are only allowed for PERSONAL and ACADEMIC usage.
