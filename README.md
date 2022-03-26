# About this work
This repo forks the original [paper](https://hszhao.github.io/papers/cvpr20_san.pdf) code and modifies it to allow training of other architectures and to better address our needs.

## Our hardware and software setup

   - Hardware setup 1: tested with two Titan X (12Gb, 24 Gb total).
   - Hardware setup 2: tested with two Titan RTX (24Gb, 48 Gb total).
   - Software: tested with PyTorch 1.9, Python3.9, CUDA toolkit 10.2, CuPy=9.4, tensorboardX. See <a href='#exampleInstall'>example installation section</a> for more requirements and install instructions.

<div id="exampleInstall">

## Example installation

```shell
git clone https://github.com/FearsomeTuna/Hybrid.git
cd Hybrid
conda create -n torchEnv
conda activate torchEnv
conda install pytorch torchvision cupy cudatoolkit=10.2 tensorboardx torchmetrics pyyaml -c pytorch -c conda-forge
pip install opencv-python
```
</div>

<div id="preparation">

## Dataset preparation

Prior to training, datasets must be already split and organized in 'train', 'val' and (possibly) 'test' sets. Our setup allows for 2 different ways to initialize the datasets at training/testing:

1. 'train' and 'val' folders with images grouped into subclasses (same as pytorch's torchvision.datasets.ImageFolder requires). Not supported for SBIR training.
2. Txt files: for each set, two files are required: one type with img_path -> target_class mappings and another with class_name -> target_class mappings. util.dataset.py for options. We provide the txt files used in the dataset folder. Root path may need to be changed. There should be a dataset/flickr25k/sketches with the same contents as dataset/Sketch_EITZ (contents can be copied or a softlink can be made to avoid duplication).

Datasets can be found on:
- [TU-Berlin (Sketch-eitz)](https://github.com/jmsaavedrar/convnet2) for a resized version. Original [here](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/)
- [QuickDraw](https://github.com/googlecreativelab/quickdraw-dataset) (may need some additional processing to get jpg images).
- [Flickr15k](https://www.cvssp.org/data/Flickr25K/cag17.html).
- Flickr25k (photos) may also be found on the previous source.
- [ImageNet (ILSVRC)](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data).

Not all sources use the same class mappings. The provided source for Flickr15k has consistent mappings between sketches and photos, but mappings between Flickr25k (photos) and TU-Berlin may not be consistent, and it may be needed to rename some folders/classes for them to match. It is recommended to check this from the start. SBIR training scripts will notify the user if class mappings don't match (or, if using the provided txt files, will crash if images are not found).

</div>

<div id='configSection'>

## Set configurations on a config file

Along with dataset info, a variety of parameters can/must be set through config YAML files. Example config files are in `config/sample` directory. Modifiying batch sizes (train, validation and test) and number of workers might be of particular interest to run implemented models on less powerful hardware. For instance, train and validation batch sizes of 32 and 16, respectively, along with 2 workers for data loaders, where used succesfully on google collab environment for pure SAN classification models on early tests (12gb vram).

Config files should be placed inside a `config/{dataset}` directory using the `{dataset}_{name}.yaml` naming convention.

</div>

## Example train 

Set the correct environment name in all .sh files in tool folder, like in the following example:

```bash
export PYTHONPATH=./
eval "$(conda shell.bash hook)"
conda activate torchEnv # set this to your conda environment name, if you use a different name
```
The following example runs training and testing for resnet26 variant.

```bash
cd Hybrid
# run both train and test for normal classification
bash tool/train.sh sketch_eitz resnet26

# run imagenet pre-train
bash tool/train.sh imagenet resnet26

# run phase 1 sbir training with imagenet pre-train weights (separate classification for each branch)
bash tool/train.sh sketch_eitz resnet26_imgnet_weights
bash tool/train.sh flickr25k_photos resnet26_imgnet_weights

# run phase 2-3 sbir training
bash too/sbir_train.sh flickr resnet26
```

To run only test, use ```tool/test.sh``` and ```tool/sbir_test.sh``` scripts.
The following example runs classification training, testing and monomodal retrieval testing on mini_quickdraw, for resnet26:

```bash
cd Hybrid
# run both train and test for normal classification
bash tool/train.sh mini_quickdraw resnet26

# monomodal retrieval test
bash tool/monomodal_zeroshot_test.sh mini_quickdraw resnet26;
```

Theoretical complexity and parameter count for all variants can be calculated through:
```bash
cd Hybrid
bash tool/run_env.sh tool/counter.py
```
```run_env.sh``` runs the next argument as a python script on the conda environment, and passes the remaining command line arguments to that script.


## See results on tensorboard
Tensorboard results take some time to fully load, even if graphic interface is already loaded, resulting in semingly incomplete graphs. Refreshing will update results as they load on the background. Having more than one events file in the same folder may cause tensorboard to incorrectly load data.
```bash
cd Hybrid
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

