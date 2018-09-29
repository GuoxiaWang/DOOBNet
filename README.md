# DOOBNet: Deep Object Occlusion Boundary Detection from an Image ([arXiv](https://arxiv.org/abs/1806.03772)) accepted by ACCV2018[Oral]

Created by Guoxia Wang.

### Introduction

Object occlusion boundary detection is a fundamental and crucial research problem in computer vision. This is challenging to solve as encountering the extreme boundary/non-boundary class imbalance during training an object occlusion boundary detector. In this paper, we propose to address this class imbalance by up-weighting the loss contribution of false negative and false positive examples with our novel Attention Loss function. We also propose a unified end-to-end multi-task deep object occlusion boundary detection network (DOOBNet) by sharing convolutional features to simultaneously predict object boundary and occlusion orientation. DOOBNet adopts an encoder-decoder structure with skip connection in order to automatically learn multi-scale and multi-level features. We significantly surpass the state-of-the-art on the PIOD dataset (ODS F-score of .702) and the BSDS ownership dataset (ODS F-score of .555), as well as improving the detecting speed to as 0.037s per image on the PIOD dataset.

### Citation

If you find DOOBNet useful in your research, please consider citing:
```
@article{wang2018doobnet,
  Title = {DOOBNet: Deep Object Occlusion Boundary Detection from an Image},
  Author = {Guoxia Wang and Xiaohui Liang and Frederick W. B. Li},
  Journal = {arXiv preprint arXiv:1806.03772},
  Year = {2018}
}
```

## Demo

Here, we assume that you locate in the DOOBNet root directory `$DOOBNET_ROOT`.

If you want to run our DOOBNet quickly, you need to download our trained model from [DOOBNet PIOD](https://drive.google.com/open?id=1suGHfi1_Leky1nwqCqbSdNRrOfqF6XNF) and save the `doobnet_piod.caffemodel` to `$DOOBNET_ROOT/examples/doobnet/Models/`. Then move to the folder and run the python demo script.

```
cd $DOOBNET_ROOT/examples/doobnet
python doobnet_demo.py
```


## Data Preparation


#### PASCAL Instance Occlusion Dataset (PIOD)

You may download the dataset original images from [PASCAL VOC 2010](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar) and annotations from [here](https://drive.google.com/file/d/0B7DaWBKShuMBSkZ6Mm5RVmg5ck0/view?usp=sharing). Then you should copy or move `JPEGImages` folder in PASCAL VOC 2010 and `Data` folder and val\_doc_2010.txt in PIOD to `data/PIOD/`. You will have the following directory structure:
```
PIOD
|_ Data
|  |_ <id-1>.mat
|  |_ ...
|  |_ <id-n>.mat
|_ JPEGImages 
|  |_ <id-1>.jpg
|  |_ ...
|  |_ <id-n>.jpg
|_ val_doc_2010.txt
```

Now, you can use data convert tool to augment and generate HDF5 format data for DOOBNet. 
```
mkdir data/PIOD/Augmentation

python doobscripts/doobnet_mat2hdf5_edge_ori.py \
--dataset PIOD \
--label-dir data/PIOD/Data \
--img-dir data/PIOD/JPEGImages \
--piod-val-list-file data/PIOD/val_doc_2010.txt \
--output-dir data/PIOD/Augmentation
```

#### BSDS ownership

For BSDS ownership dataset, you may download the dataset original images from [BSDS300](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz) and annotations from [here](https://drive.google.com/open?id=0B7DaWBKShuMBd3Z0Vmk3UkZxcUU). Then you should copy or move `BSDS300` folder in BSDS300-images and `trainfg` and `testfg` folder in BSDS\_theta to `data/BSDSownership/`. And you will have the following directory structure:
```
BSDSownership
|_ trainfg
|  |_ <id-1>.mat
|  |_ ...
|  |_ <id-n>.mat
|_ testfg
|  |_ <id-1>.mat
|  |_ ...
|  |_ <id-n>.mat
|_ BSDS300
|  |_ images
|     |_ train
|        |_ <id-1>.jpg
|        |_ ...
|        |_ <id-n>.jpg
|     |_ ...
|  |_ ...
```
Note that BSDS ownership's test set are split from 200 train images (100 for train, 100 for test). More information you can check ids in `trainfg` and `testfg` folder and ids in `BSDS300/images/train` folder, or refer to [here](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/fg/fgdata.tar.gz)

Run the following code for BSDS ownership dataset. 
```
mkdir data/BSDSownership/Augmentation

python doobscripts/doobnet_mat2hdf5_edge_ori.py \
--dataset BSDSownership \
--label-dir data/BSDSownership/trainfg \
--img-dir data/BSDSownership/BSDS300/images/train \
--bsdsownership-testfg data/BSDSownership/testfg \
--output-dir data/BSDSownership/Augmentation 
```

## Training

Firstly, you need to download the Res50 weight file from [Res50](https://drive.google.com/open?id=1nyGjqSj0LGVsY9iBhsEdo-TXSyROGTgZ) and save `resnet50.caffemodel` to the folder `$DOOBNET_ROOT/models/resnet/`.

#### PASCAL Instance Occlusion Dataset (PIOD)

For training DOOBNet on PIOD training dataset, you can run:

```
cd $DOOBNET_ROOT/examples/doobnet/PIOD

./train.sh
```
When training completed, you need to modify `model = '../Models/doobnet_piod.caffemodel'` in `deploy_doobnet_piod.py` and then run `python deploy_doobnet_piod.py` to get the results on PIOD testing dataset. For comparation, you can also download our trained model from [DOOBNet PIOD](https://drive.google.com/open?id=1suGHfi1_Leky1nwqCqbSdNRrOfqF6XNF).


#### BSDS ownership
For training DOOBNet on BSDS ownership, you can refer the manner as same as PIOD dataset above. You can download trained DOOBNet on BSDS ownership from here [DOOBNet BSDSownership](https://drive.google.com/open?id=1TMocxt9iE9w6qVLo4iFVwHoQm9nbsX9K).


## Evaluation

Here we provide the PIOD and the BSDS ownership dataset's evaluation and visualization code in `doobscripts` folder.

**Note that** you need to config the necessary paths or variables. More information please refers to `doobscripts/README.md`.

To run the evaluation:
```
run doobscripts/evaluation/EvaluateOcc.m
```

#### Option
For visualization, to run the script:
```
run doobscripts/visulation/PlotAll.m
```
