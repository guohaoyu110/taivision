> # **Taivision Library**

![PyTorch Logo](https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png)

![data](WechatIMG67.png)


<!-- toc -->

- [Pretrained Models](#pretrained-models)
  - [Model Comparison Chart](#model-comparison-chart) 
  - [Gluoncv: How To Use Pretrained Models](#gluoncv-how-to-use-pretrained-models)
  - [PyTorch: How To Use Pretrained Models](#pytorch-how-to-use-pretrained-models)
- [Datasets](#datasets)
  - [Torchvision Datasets](#torchvision-datasets)
  - [Gluoncv Datasets](#gluoncv-datasets)
  - [Dataset Comparison Chart](#dataset-comparison-chart)
  - [PyTorch Dataloader](#pytorch-dataloader)
  - [Gluoncv Dataloader](#gluoncv-dataloader)
  - [Create Custom Datasets for Object Detection on Gluoncv](#create-custom-datasets-for-object-detection-on-gluoncv)
  - [Create Custom Datasets on PyTorch](#create-custom-datasets-on-pytorch)
<!-- tocstop -->




# Pretrained Models



## Model Comparison Chart

model name         | gluoncv          | torchvision    |
--------------------|------------------|-----------------------|
Resnet	(classification)			| 'resnet18\_v1', 'resnet34\_v1', 'resnet50\_v1', 'resnet101\_v1', 'resnet152\_v1', 'resnet18\_v2', 'resnet34_v2', 'resnet50\_v2', 'resnet101\_v2', 'resnet152\_v2', 'resnest200', 'resnest269', 'se\_resnet18\_v1', 'se\_resnet34\_v1', 'se\_resnet50\_v1', 'se\_resnet101\_v1', 'se\_resnet152\_v1', 'se\_resnet18\_v2', 'se\_resnet34\_v2', 'se\_resnet50\_v2', 'se\_resnet101\_v2', 'se\_resnet152\_v2'   |  'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'  | 
Resnext (classification)|'resnext50\_32x4d', 'resnext101\_32x4d', 'resnext101\_64x4d', 'resnext101e\_64x4d', 'se\_resnext50\_32x4d', 'se\_resnext101\_32x4d', 'se\_resnext101\_64x4d', 'se\_resnext101e\_64x4d' |'ResNeXt-50-32x4d', 'ResNeXt-101-32x8d'
Inception (classification)| 'inceptionv3'| 'Inception v3'
VGG (classification)| 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11\_bn', 'vgg13\_bn', 'vgg16\_bn', 'vgg19\_bn' |VGG-11, VGG-13, VGG-16, VGG-19
Alexnet	(classifcation)  | 'alexnet'  | 'Alexnet'   |
squeezenet (classifcation） | 'squeezenet1.0', 'squeezenet1.1'  | 'SqueezeNet 1.0', 'SqueezeNet 1.1'  |
densenet		       | 'densenet121', 'densenet161', 'densenet169', 'densenet201'  |   |
googlenet	       | 'googlenet'  |   |
xception		       | 'xception', 'xception71'  |  |
mobilenet		       | 'mobilenet1.0', 'mobilenet0.75', 'mobilenet0.5', 'mobilenet0.25', 'mobilenetv2\_1.0', 'mobilenetv2\_0.75', 'mobilenetv2\_0.5', 'mobilenetv2\_0.25', 'mobilenetv3\_large', 'mobilenetv3\_small', 'mobile\_pose\_resnet18\_v1b', 'mobile\_pose\_resnet50\_v1b', 'mobile\_pose\_mobilenet1.0', 'mobile\_pose\_mobilenetv2\_1.0', 'mobile\_pose\_mobilenetv3\_large', 'mobile\_pose\_mobilenetv3\_small'  |  |
ssd		       | 'ssd\_300\_vgg16\_atrous\_voc', 'ssd\_300\_vgg16\_atrous\_coco', 'ssd\_300\_vgg16\_atrous\_custom', 'ssd\_512\_vgg16\_atrous\_voc', 'ssd\_512\_vgg16\_atrous\_coco', 'ssd\_512\_vgg16\_atrous\_custom', 'ssd\_512\_resnet18\_v1\_voc', 'ssd\_512\_resnet18\_v1\_coco', 'ssd\_512\_resnet50\_v1\_voc', 'ssd\_512\_resnet50\_v1\_coco', 'ssd\_512\_resnet50\_v1\_custom', 'ssd\_512\_resnet101\_v2\_voc', 'ssd\_512\_resnet152\_v2\_voc', 'ssd\_512\_mobilenet1.0\_voc', 'ssd\_512\_mobilenet1.0\_coco', 'ssd\_300\_mobilenet1.0\_lite\_coco', 'ssd\_512\_mobilenet1.0\_custom', 'ssd\_300\_mobilenet0.25\_voc', 'ssd\_300\_mobilenet0.25\_coco', 'ssd\_300\_mobilenet0.25\_custom', 'ssd\_300\_resnet34\_v1b\_voc', 'ssd\_300\_resnet34\_v1b\_coco', 'ssd\_300\_resnet34\_v1b\_custom'  |   |
faster rcnn		       |'faster\_rcnn\_resnet50\_v1b\_voc', 'mask\_rcnn\_resnet18\_v1b\_coco', 'faster\_rcnn\_resnet50\_v1b\_coco', 'faster\_rcnn\_fpn\_resnet50\_v1b\_coco', 'faster\_rcnn\_fpn\_syncbn\_resnet50\_v1b\_coco', 'faster\_rcnn\_fpn\_syncbn\_resnest50\_coco', 'faster\_rcnn\_resnet50\_v1b\_custom', 'faster\_rcnn\_resnet101\_v1d\_voc', 'faster\_rcnn\_resnet101\_v1d\_coco', 'faster\_rcnn\_fpn\_resnet101\_v1d\_coco', 'faster\_rcnn\_fpn\_syncbn\_resnet101\_v1d\_coco', 'faster\_rcnn\_fpn\_syncbn\_resnest101\_coco', 'faster\_rcnn\_resnet101\_v1d\_custom', 'faster\_rcnn\_fpn\_syncbn\_resnest269\_coco', 'custom\_faster\_rcnn\_fpn'  |   |
mask rcnn	       | 'mask\_rcnn\_resnet18\_v1b\_coco',  'mask\_rcnn\_resnet50\_v1b\_coco', 'mask\_rcnn\_fpn\_resnet50\_v1b\_coco', 'mask\_rcnn\_resnet101\_v1d\_coco', 'mask\_rcnn\_fpn\_resnet101\_v1d\_coco', 'mask\_rcnn\_fpn\_resnet18\_v1b\_coco', 'mask\_rcnn\_fpn\_syncbn\_resnet18\_v1b\_coco', 'mask\_rcnn\_fpn\_syncbn\_mobilenet1\_0\_coco', 'custom\_mask\_rcnn\_fpn'  |   
cifar resnet| 'cifar\_resnet20\_v1', 'cifar\_resnet56\_v1', 'cifar\_resnet110\_v1', 'cifar\_resnet20\_v2', 'cifar\_resnet56\_v2', 'cifar\_resnet110\_v2', 'cifar\_wideresnet16\_10', 'cifar\_wideresnet28\_10', 'cifar\_wideresnet40\_8', 'cifar\_resnext29\_32x4d', 'cifar\_resnext29\_16x64d'|
fcn resnet |'fcn\_resnet50\_voc', 'fcn\_resnet101\_coco', 'fcn\_resnet101\_voc', 'fcn\_resnet50\_ade', 'fcn\_resnet101\_ade'|
psp resnet|'psp\_resnet101\_coco', 'psp\_resnet101\_voc', 'psp\_resnet50\_ade', 'psp\_resnet101\_ade', 'psp\_resnet101\_citys'|
deeplab | 'deeplab\_resnet101\_coco', 'deeplab\_resnet101\_voc', 'deeplab\_resnet152\_coco', 'deeplab\_resnet152\_voc', 'deeplab\_resnet50\_ade', 'deeplab\_resnet101\_ade', 'deeplab\_resnest50\_ade', 'deeplab\_resnest101\_ade', 'deeplab\_resnest200\_ade', 'deeplab\_resnest269\_ade', 'deeplab\_resnet50\_citys', 'deeplab\_resnet101\_citys', 'deeplab\_v3b\_plus\_wideresnet\_citys'|
icnet |'icnet\_resnet50\_citys', 'icnet\_resnet50\_mhpv1'  |
danet|'danet\_resnet50\_citys', 'danet\_resnet101\_citys'|
resnet(姿态估计）|'resnet18\_v1b', 'resnet34\_v1b', 'resnet50\_v1b', 'resnet50\_v1b\_gn', 'resnet101\_v1b\_gn', 'resnet101\_v1b', 'resnet152\_v1b', 'resnet50\_v1c', 'resnet101\_v1c', 'resnet152\_v1c', 'resnet50\_v1d', 'resnet101\_v1d', 'resnet152\_v1d', 'resnet50\_v1e', 'resnet101\_v1e', 'resnet152\_v1e', 'resnet50\_v1s', 'resnet101\_v1s', 'resnet152\_v1s'|
yolo | 'darknet53', 'yolo3\_darknet53\_coco', 'yolo3\_darknet53\_voc', 'yolo3\_darknet53\_custom', 'yolo3\_mobilenet1.0\_coco', 'yolo3\_mobilenet1.0\_voc', 'yolo3\_mobilenet1.0\_custom', 'yolo3\_mobilenet0.25\_coco', 'yolo3\_mobilenet0.25\_voc', 'yolo3\_mobilenet0.25\_custom'|
nasnet |'nasnet\_4\_1056', 'nasnet\_5\_1538', 'nasnet\_7\_1920', 'nasnet\_6\_4032'|
simple pose| 'simple\_pose\_resnet18\_v1b', 'simple\_pose\_resnet50\_v1b', 'simple\_pose\_resnet101\_v1b', 'simple\_pose\_resnet152\_v1b', 'simple\_pose\_resnet50\_v1d', 'simple\_pose\_resnet101\_v1d', 'simple\_pose\_resnet152\_v1d'|
残差注意力网络|'residualattentionnet56', 'residualattentionnet92', 'residualattentionnet128', 'residualattentionnet164', 'residualattentionnet200', 'residualattentionnet236', 'residualattentionnet452', 'cifar\_residualattentionnet56', 'cifar\_residualattentionnet92', 'cifar\_residualattentionnet452'|
kinetic resnet |'resnet18\_v1b\_kinetics400\_int8', 'resnet50\_v1b\_kinetics400\_int8', 'inceptionv3\_kinetics400\_int8', 'inceptionv3\_kinetics400', 'inceptionv3\_sthsthv2', 'c3d\_kinetics400', 'p3d\_resnet50\_kinetics400', 'p3d\_resnet101\_kinetics400', 'r2plus1d\_resnet18_kinetics400', 'r2plus1d\_resnet34_kinetics400', 'r2plus1d\_resnet50_kinetics400', 'r2plus1d\_resnet101\_kinetics400', 'r2plus1d\_resnet152\_kinetics400', 'i3d\_resnet50\_v1\_ucf101', 'i3d\_resnet50\_v1\_hmdb51', 'i3d\_resnet50\_v1\_kinetics400', 'i3d\_resnet50\_v1\_sthsthv2', 'i3d\_resnet50\_v1\_custom', 'i3d\_resnet101\_v1\_kinetics400', 'i3d\_inceptionv1\_kinetics400', 'i3d\_inceptionv3\_kinetics400', 'i3d\_nl5\_resnet50\_v1\_kinetics400', 'i3d\_nl10\_resnet50\_v1\_kinetics400', 'i3d\_nl5\_resnet101\_v1\_kinetics400', 'i3d\_nl10\_resnet101\_v1\_kinetics400', 'slowfast\_4x16\_resnet50\_kinetics400', 'slowfast\_4x16\_resnet50\_custom', 'slowfast\_8x8\_resnet50\_kinetics400', 'slowfast\_4x16\_resnet101\_kinetics400', 'slowfast\_8x8\_resnet101\_kinetics400', 'slowfast\_16x8\_resnet101\_kinetics400', 'slowfast\_16x8\_resnet101\_50\_50\_kinetics400', 'resnet18\_v1b\_kinetics400', 'resnet34\_v1b\_kinetics400', 'resnet50\_v1b\_kinetics400', 'resnet101\_v1b\_kinetics400', 'resnet152\_v1b\_kinetics400'|
fcn-resnet |fcn\_resnet101\_voc\_int8', 'fcn\_resnet101\_coco\_int8'|





## Gluoncv: How To Use Pretrained Models
The following example requires GluonCV>=0.4 and MXNet>=1.4.0. Please follow our installation guide to install or upgrade GluonCV and MXNet if necessary.

Prepare an image by yourself or use our sample image. You can save the image into filename classification-demo.png in your working directory or change the filename in the source codes if you use an another name.

Use a pre-trained model. A model is specified by its name.

```
net = getmodel(model_name, class = XXX, pretrained = True)

(with pretrained weights)
```

## PyTorch: How To Use Pretrained Models

[https://pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html)

```
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
```


# Datasets

## Torchvision Datasets
```
## torchvision
e.g trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
```


## Gluoncv Datasets


[https://cv.gluon.ai/build/examples_datasets/ade20k.html](https://cv.gluon.ai/build/examples_datasets/ade20k.html)

[https://cv.gluon.ai/api/data.datasets.html](https://cv.gluon.ai/api/data.datasets.html)

[https://mxnet.apache.org/versions/1.7.0/api/python/docs/api/gluon/data/vision/datasets/index.html](https://mxnet.apache.org/versions/1.7.0/api/python/docs/api/gluon/data/vision/datasets/index.html)

- Some need to be downloaded, some don't need and just add library from datasets





## Dataset Comparison Chart
- S represents support, N reprents not support

dataset name         | gluoncv          | torchvision    |
--------------------|------------------|-----------------------
ADE20K |S | N
CelebA |  N | S
MNIST|S| S
CIFAR10 |S| S
CIFAR100| S| S
Cityscapes|S (data.CitySegmentation)| S
COCO Captions|N| S
COCO Detection| S (data.COCODetection)| S
DatasetFolder|N| S
EMNIST |N| S
FakeData|N| S
Fasion-MNIST| S | S
Flickr|N| S
HMDB51|N| S
ImageFolder|S (datasets.ImageFolderDataset)| S
ImageRecordDataset | S (datasets.ImageRecordDataset)| N
Imagenet| S (data.ImageNet)| S
Kinetics-400|S (data.Kinetics400)| S
KMNIST|N| S
LSUN|N| S
Omniglot|N| S
PhotoTour|N| S
Places365|N| S
QMNIST|N| S
SBD|N| S
SBU|N| S
STL10|N| S
SVHN|N| S
UCF101|S  (data.UCF101)| S
USPS|N| S
ILSVRC 2015 DET dataset |S | N
ILSVRC 2015 VId dataset |S| N
Multi-Human Parsing V1 dataset |S| N
OTB 2015 dataset  |S| N 
PASCAL VOC datasets |S(data.VOCDetection or data.VOCSegmentation)|  S 
Youtube_bb dataset | S | N
20BN-something-something Dataset V2 |S| N
HMDB51 Dataset | S (data.HMDB51)| N



## PyTorch Dataloader
```
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
shuffle=True, num_workers=2)
```

## Gluoncv Dataloader
```
class mxnet.gluon.data.DataLoader(dataset, batch_size=None, shuffle=False, 
sampler=None, last_batch=None, batch_sampler=None, 
batchify_fn=None, num_workers=0, pin_memory=False, 
pin_device_id=0, prefetch=None, thread_pool=False, timeout=120)
```


## Create Custom Datasets for Object Detection on Gluoncv

1. Create a folder, name it "myDataset".

2. Inside the folder you just created, create another folder, this time call it "VOC2019".

3. Inside the folder you just created (yeah, again), create three folders: "Annotations", "ImageSets" and "JPEGImages".

4. Inside the folder Annotations, put the annotation files structured as in the link you mentioned, while in the folder JPEGImages, put the images in JPEG format. Note that the image name corresponds to the annotation file with the same name.

5. In the folder ImageSets, create a folder (yep, another one) name "Main".

6. Inside the folder you just created, create a file "train.txt" with a list of names (one per line) corresponding to the images you want to be used for training (it will be taken from the folders Annotations and JPEGImages).

```
classes = ['class1', class2', 'class3']

class VOCLike(VOCDetection):
    CLASSES = classes

    def __init__(self, root, splits, transform=None, index_map=None, preload_label=True):
        super(VOCLike, self).__init__(root, splits, transform, index_map, preload_label)

voc_dataset =VOCLike(root='/my/path/myDataset', splits=[(2019, 'train')])

```

## Create Custom Datasets on PyTorch

[https://pytorch.org/tutorials/beginner/data\_loading\_tutorial.html](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)






