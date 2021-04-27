# Forked from original paper code.
This repo takes the original paper code and modifies it to allow train of resnet26 models.

## Example installation

```
git clone https://github.com/FearsomeTuna/SAN.git
cd SAN
conda update -n base -c defaults conda
conda create -n torchEnv
conda activate torchEnv
conda install pytorch=1.8.1 torchvision=0.9.1 cupy=9.0.0 cudatoolkit=10.1 pyyaml tensorboardx -c pytorch -c conda-forge
pip install git+https://github.com/FrancescoSaverioZuppichini/glasses@1dbe60f99c68f8b65efb7d70fe829623fdb4a689
pip install opencv-python requests==2.23.0 matplotlib einops rich torchinfo
```
## Download and prepare datasets

Datasets (or a symbolic link to them) must be located in a 'dataset' folder on SAN root, organized in one of 3 ways:

- One 'data' folder: data is sampled into train and val during runtime.
- 'train' and 'test' folders: data for train and validation is sampled from 'train' folder at runtime.
- 'train', 'val' and 'test' folders.
Keep in mind 'test' folder is not needed for training, so you can make do without it as long as you do not run `test.sh` file.

Independently of which of these 3 configurations are used, images within must be organized into subfolders for each class.

Dataset organization must be declared in config file for correct dataset import (see next sample config file in config directory and <a href='#configSection'>config</a> section for more info).

<div id='configSection'>

## Set configurations on a config file

Along with dataset organization, a variety of experimental parameters can/must be set through config files in the config directory. Along with dataset organization, modifiying configs for batch sizes (train, validation and test) might be of interest to run implemented models on less powerful hardware (train and validation batch sizes of 32 and 16, respectively, where used succesfully on collab environment).

Config files should be placed in the `config` directory using the `{dataset}_{name}.yaml` naming convention.

</div>

## Example dataset preparation
Sketch_EITZ and mnist download links can be found on [this](https://github.com/jmsaavedrar/convnet2) repo.EITZ has data in a single folder with all data organized into subfolders for each class already. MNIST is spread in two download links and is compressed in a somewhat cumbersome way. This code can be used to unzip and prepare them:

```shell
SAN_ROOT="/path/to/SAN" # SAN root
DOWNLOAD_PATH="/path/to/downloaded/files" # download directory

cd $SAN_ROOT
mkdir dataset

# EITZ
unzip $DOWNLOAD_PATH/EITZ.zip -d dataset
mv dataset/Sketch_EITZ/png_w256 dataset/Sketch_EITZ/data

# MNIST
cd dataset
mkdir MNIST
cd MNIST
mv $DOWNLOAD_PATH/mnist_train.gzip mnist_train.gz
mv $DOWNLOAD_PATH/mnist_test.gzip mnist_test.gz
gunzip mnist_train.gz
gunzip mnist_test.gz
tar -xvf mnist_train
tar -xvf mnist_test
rm mnist_train mnist_test
mv Test val
mv Train train

# group mnist images into subfolders for classes 0 to 9
cd val
for i in {0..9}; do mkdir $i; mv *_$i.png $i; done

cd ../train
for i in {0..9}; do mkdir $i; mv *_$i.png $i; done

cd $SAN_ROOT
```

## Example use

Set the correct environment name in tool/train.sh

```
export PYTHONPATH=./
eval "$(conda shell.bash hook)"
conda activate torchEnv
```

```
cd SAN
sh tool/train.sh sketch_eitz resnet26
```

## See results on tensorboard

```
cd SAN
tensorboard --logdir exp
```

## Google Collab Demo
An example use can be found on https://colab.research.google.com/drive/1iq6aeMK6lryka6WYimRszTy3uT44h8Ts#scrollTo=Dh5iyYNLh1ys. This is based on an older commit, with different training configs to adjust to collab gpu.

# Original readme follows:

# Exploring Self-attention for Image Recognition
by Hengshuang Zhao, Jiaya Jia, and Vladlen Koltun, details are in [paper](https://hszhao.github.io/papers/cvpr20_san.pdf).

### Introduction

This repository is build for the proposed self-attention network (SAN), which contains full training and testing code. The implementation of SA module with optimized CUDA kernels are also included.

<p align="center"><img src="./figure/sa.jpg" width="400"/></p>

### Usage

1. Requirement:

   - Hardware: tested with 8 x Quadro RTX 6000 (24G).
   - Software: tested with PyTorch 1.4.0, Python3.7, CUDA 10.1, [CuPy](https://cupy.chainer.org/) 10.1, [tensorboardX](https://github.com/lanpa/tensorboardX).

2. Clone the repository:

   ```shell
   git clone https://github.com/hszhao/SAN.git
   ```
   
3. Train:

   - Download and [prepare](https://github.com/facebookarchive/fb.resnet.torch/blob/master/INSTALL.md) the ImageNet dataset (ILSVRC2012) and symlink the path to it as follows (you can alternatively modify the relevant path specified in folder `config`):

     ```
     cd SAN
     mkdir -p dataset
     ln -s /path_to_ILSVRC2012_dataset dataset/ILSVRC2012
     ```

   - Specify the gpus (usually 8 gpus are adopted) used in config and then do training:

     ```
     sh tool/train.sh imagenet san10_pairwise
     ```
     
   - If you are using [SLURM](https://slurm.schedmd.com/documentation.html) for nodes manager, uncomment lines in `train.sh` and then do training:

     ```shell
     sbatch tool/train.sh imagenet san10_pairwise
     ```

4. Test:

   - Download trained SAN [models](https://drive.google.com/open?id=19ZJn48UpiF_j8e1-UU9UZ4AVfLjmwPzH) and put them under folder specified in config or modify the specified paths, and then do testing:

     ```shell
     sh tool/test.sh imagenet san10_pairwise
     ```
   
5. Visualization:

   - [tensorboardX](https://github.com/lanpa/tensorboardX) incorporated for better visualization regarding curves:

     ```shell
     tensorboard --logdir=exp/imagenet
     ```

6. Other:

   - Resources: GoogleDrive [LINK](https://drive.google.com/open?id=19ZJn48UpiF_j8e1-UU9UZ4AVfLjmwPzH) contains shared models.

### Performance

Train Parameters: train_gpus(8), batch_size(256), epochs(100), base_lr(0.1), lr_scheduler(cosine), label_smoothing(0.1), momentum(0.9), weight_decay(1e-4).

Overall result:

| Method       | top-1 | top-5 | Params | Flops |
| :----------- | :---: | :---: | :----: | :---: |
| ResNet26     | 73.6  | 91.7  | 13.7M  | 2.4G  |
| SAN10-pair.  | 74.9  | 92.1  | 10.5M  | 2.2G  |
| SAN10-patch. | 77.1  | 93.5  | 11.8M  | 1.9G  |
| ResNet38     | 76.0  | 93.0  | 19.6M  | 3.2G  |
| SAN15-pair.  | 76.6  | 93.1  | 14.1M  | 3.0G  |
| SAN15-patch. | 78.0  | 93.9  | 16.2M  | 2.6G  |
| ResNet50     | 76.9  | 93.5  | 25.6M  | 4.1G  |
| SAN19-pair.  | 76.9  | 93.4  | 17.6M  | 3.8G  |
| SAN19-patch. | 78.2  | 93.9  | 20.5M  | 3.3G  |

### Citation

If you find the code or trained models useful, please consider citing:

```
@inproceedings{zhao2020san,
  title={Exploring Self-attention for Image Recognition},
  author={Zhao, Hengshuang and Jia, Jiaya and Koltun, Vladlen},
  booktitle={CVPR},
  year={2020}
}
```

