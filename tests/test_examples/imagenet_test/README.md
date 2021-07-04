# ImageNet training in PyTorch

This implements training of popular model architectures, such as ResNet, AlexNet, and VGG on the ImageNet dataset.

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset and move validation images to labeled subfolders
    - To do this, you can use the following script: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh


## How to prepare dataset

You need to download ILSVRC2012 dataset, and make sure the folder architecture as follows:

```
ILSVRC2012
|
|-----train----1000 sub classes folders
|
|-----val------1000 sub classes folders
Please make sure the same class has same class folder name in train and val folders.
```



## Training

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
python main.py -a resnet18 [imagenet-folder with train and val folders]
```

The default learning rate schedule starts at 0.1 and decays by a factor of 10 every 30 epochs. This is appropriate for ResNet and models with batch normalization, but too high for AlexNet and VGG. Use 0.01 as the initial learning rate for AlexNet or VGG:

```bash
python main.py -a alexnet --lr 0.01 [imagenet-folder with train and val folders]
```

## ILSVRC2012 classification training results 

### Training in nn.parallel model results 

| Network       | gpu-num | warm up | lr decay | total epochs | Top-1 error |
| --- | --- |  --- |  --- |  --- |  --- | 
| ResNet-18     | 4 RTX2080Ti | no | multistep | 100 | 29.684 | 
| ResNet-34     | 4 RTX2080Ti | no | multistep | 100 | 26.264 | 
| ResNet-50     | 4 RTX2080Ti | no | multistep | 100 | 23.488 | 
| ResNet-101    | 4 RTX2080Ti | no | multistep | 100 | 22.276 | 

All nets are trained by input_size=224*224.

For training resnet50 with batch_size=256, you need at least 4 2080Ti GPUs and need about 3 or 4 days. 

# ImageNet Test in PyTorch 

## Requirements 

Platform: Ubuntu 16

```
python==3.7.7
torch==1.8.0
torchvision==0.9.0
torchaudio==0.8.0
pycocotools==2.0.2
numpy
Cython
matplotlib
opencv-python
tqdm
thop
```

## How to instll apex?
apex needs to be installed separately. Please use the following orders to install apex. 

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

If the above command fails to install apex, you can use the following orders to install apex:

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
```


Using apex to train can reduce video memory usage by 25%-30%, but the training speed will be slower, the trained model has the same performance as not using apex. 

## test_scripts 
```
bash test.sh 
```


## inference_demo 

```
bash run_classify_single_image.sh
```
