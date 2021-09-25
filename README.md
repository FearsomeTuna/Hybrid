# About this work
This repo forks the original [paper](https://hszhao.github.io/papers/cvpr20_san.pdf) code and modifies it to allow training of other architectures and to better address our needs.

## Our hardware and software setup

   - Hardware: tested with two Titan X (12Gb, 24 Gb total).
   - Software: tested with PyTorch 1.9, Python3.9, CUDA toolkit 10.2, CuPy=9.4, tensorboardX. See <a href='#exampleInstall'>example installation section</a> for more requirements and install instructions.

<div id="exampleInstall">

## Example installation

```shell
git clone https://github.com/FearsomeTuna/SAN.git
cd SAN
conda create -n torchEnv
conda activate torchEnv
conda install pytorch torchvision cupy cudatoolkit=10.2 tensorboardx torchmetrics pyyaml pandas -c pytorch -c conda-forge
pip install opencv-python
```
</div>

<div id="preparation">

## Dataset preparation

Prior to training, datasets must be already split and organized in 'train', 'val' and (possibly) 'test' sets. Our setup allows for 3 different ways to initialize the datasets at training/testing:

1. 'train' and 'val' folders with images grouped into subclasses (same as pytorch's torchvision.datasets.ImageFolder requires).
2. Custom 'train_init.pt' and 'val_init.pt' files with list of predefined (image_path, target_class) tuples. This can be achieved with make_dataset_path.py and split_dataset.py utilities.
3. Custom 'train_imgs.pt' and 'val_imgs.pt' files with byte images ready to be loaded whole into memory. This can be achieved with make_inmemory_dataset.py utility (which takes custom paths file as input).

These custom files are based on torch.save and torch.load functions.

Dataset organization must be declared in config file for correct dataset import (see <a href='#configSection'>config</a> section for more info).

</div>

<div id='examplePreparation'>

## Example dataset preparation

Sketch_EITZ dataset (better known as TU_Berlin in the literature) can be found [here](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/), though for our purposes, we use a lower size version found on [this](https://github.com/jmsaavedrar/convnet2) repo.
The following script generates custom files with path information and also custom byte image files from TU_Berlin dataset, saving them to folders within `SAN/dataset` folder. To run generic python files inside conda environment, we use `run_env.sh` script. This just activates conda environment and passes remaining arguments to python file:

```shell
# downloaded file: EITZ.zip from second link
DOWNLOAD_PATH="/path/to/downloaded/files" # download directory

SAN_ROOT="/path/to/SAN"
cd $SAN_ROOT
mkdir dataset

# EITZ
unzip $DOWNLOAD_PATH/EITZ.zip -d dataset
rm dataset/Sketch_EITZ/mapping.txt dataset/Sketch_EITZ/test.txt dataset/Sketch_EITZ/train.txt
mv dataset/Sketch_EITZ/png_w256 dataset/Sketch_EITZ/data

# conda environment must be correctly setup to run utilities

# first we make paths file. Last 3 arguments are input folder (images), output folder for generated file, and output filename.
sh tool/run_env.sh tool/dataset_tool/make_dataset_paths.py dataset/Sketch_EITZ/data dataset/Sketch_EITZ data_init.pt

# take dataset/Sketch_EITZ/data_init.pt info and split it in two sets according to 0.8 and 0.2 ratios. Output to dataset/Sketch_EITZ
sh tool/run_env.sh tool/dataset_tool/split_dataset.py dataset/Sketch_EITZ/data_init.pt dataset/Sketch_EITZ -r 0.8

# rename output files
mv dataset/Sketch_EITZ/split_0.pt dataset/Sketch_EITZ/train_init.pt
mv dataset/Sketch_EITZ/split_1.pt dataset/Sketch_EITZ/val_init.pt

# make inmemory versions
sh tool/run_env.sh tool/dataset_tool/make_inmemory_dataset.py dataset/Sketch_EITZ/train_init.pt dataset/Sketch_EITZ/train_imgs.pt -t train
sh tool/run_env.sh tool/dataset_tool/make_inmemory_dataset.py dataset/Sketch_EITZ/val_init.pt dataset/Sketch_EITZ/val_imgs.pt -t val

rm dataset/Sketch_EITZ/data_init.pt
```
</div>

<div id='configSection'>

## Set configurations on a config file

Dataset location must be specified in config files. Example config files supposes a 'dataset' folder in SAN root (see `sample.yaml` config file in `config/sample` directory).

Along with dataset organization and location, a variety of parameters can/must be set through config YAML files. Modifiying batch sizes (train, validation and test) and number of workers might be of particular interest to run implemented models on less powerful hardware. For instance, train and validation batch sizes of 32 and 16, respectively, along with 2 workers for data loaders, where used succesfully on google collab environment for pure SAN classification models on early tests (12gb vram).

Config files should be placed inside a `config/{dataset}` directory using the `{dataset}_{name}.yaml` naming convention.

</div>

## Example train 

Set the correct environment name in tool/train.sh

```shell
export PYTHONPATH=./
eval "$(conda shell.bash hook)"
conda activate torchEnv # set this to your conda environment name
```
The following example runs training and testing using `sketch_eitz_resnet26.yaml` config file located in `config/sketch_eitz` folder.

```shell
cd SAN
sh tool/train.sh sketch_eitz resnet26
```

## See results on tensorboard
Tensorboard results take some time to fully load, even if graphic interface is already loaded, resulting in semingly incomplete graphs. Refreshing will update results as they load on the background.
```shell
cd SAN
tensorboard --logdir exp
```

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

