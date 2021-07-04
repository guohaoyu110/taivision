# Demo of Pre-trained Models on ImageNet

ImageNet is a large labeled dataset of real-world images. It is one of the most widely used dataset in latest computer vision research. 

<p align="center"><img src="./imagenet_mosaic.jpeg" alt="imagenet_mosaic" width="500"/></p>

In this tutorial, we will show how a pre-trained neural network
classifies real world images.

For your convenience, we provide a script that loads a pre-trained ``ResNet50`` model, and classifies an input image.

## Inference demo 


A model trained on ImageNet can classify images into 1000 classes, this makes it much more powerful than the one we shoed in the CIFAR10 demo.

With this script, you can load a pre-trained model and classify any image you have.

Let's test with the photo of horse and child photo.

<p align="center"><img src="./classification_result.jpg" alt="imagenet_mosaic" width="500"/></p>


```shell
bash run_classify_single_image.sh
```

And the model predicts that 

```
args: Namespace(input_image_size=224, model='resnet50', save_image_path='./', seed=0, show_image=True, test_image_path='../resources/detection/images/000000001551.jpg', trained_model_path='/home/xuel/xw/guohaoyu/taivision/data/resnet/resnet50-epoch100-acc76.512.pth', trained_num_classes=1000, use_gpu=False)
model: resnet50, flops: 4.112G, params: 25.557M
score: 0.784, class: 834, color: [249, 60, 60]
```

Class 834 is `suit, suit of clothes` in ImageNet dataset, so it's basically true.