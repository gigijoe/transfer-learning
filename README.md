# Pytorch Transfer Learning

This project is base on https://github.com/kuangliu/pytorch-cifar

The major functions as follow \
1.Do transfer learning base on model ResNet50 with custom dataset \
2.Inference by TensorRT \
3.Inference by TensorRT C++ API \

## Getting started

Clone from github
```
git clone https://github.com/gigijoe/transfer-learning.git
cd transfer-learning/
git submodule update --init --recursive

```

Go to folder ./pytorch-cifar and change model to ResNet50
```
cd pytorch-cifar/
vi main.py
``` 

ResNet50
```
@@ -55,7 +55,7 @@
 # Model
 print('==> Building model..')
 # net = VGG('VGG19')
-# net = ResNet18()
+net = ResNet50()
 # net = PreActResNet18()
 # net = GoogLeNet()
 # net = DenseNet121()
@@ -68,7 +68,7 @@
 # net = ShuffleNetV2(1)
 # net = EfficientNetB0()
 # net = RegNetX_200MF()
-net = SimpleDLA()
+# net = SimpleDLA()
 net = net.to(device)
 if device == 'cuda':
     net = torch.nn.DataParallel(net)

```

Start training with
```
python main.py
```
The trained model will be stored to ./checkpoint/ckpt.pth

If there's CUDA out of memory error, try reduce batch_size
```
RuntimeError: CUDA out of memory. Tried to allocate 64.00 MiB (GPU 0; 3.95 GiB total capacity; 3.20 GiB already allocated; 7.12 MiB free; 3.29 GiB reserved in total by PyTorch)
```

@@ -40,14 +40,14 @@
 ])
 
 trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
 trainloader = torch.utils.data.DataLoader(
-    trainset, batch_size=64, shuffle=True, num_workers=2)
+    trainset, batch_size=128, shuffle=True, num_workers=2)
 
 testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
 testloader = torch.utils.data.DataLoader(
-    testset, batch_size=50, shuffle=False, num_workers=2)
+    testset, batch_size=100, shuffle=False, num_workers=2)

## Transfer learning

Go back to root folder
```
cd ../
```

The custom train dataset (pictures) are in the sub folder data/ \
There are only two classes, Glider and Paraglider.
```
$ tree -d data
data
├── test
│   ├── glider
│   └── paraglider
├── train
│   ├── glider
│   └── paraglider
└── valid
    ├── glider
    └── paraglider
```

Start transfer learning
```
python3 transfer_learning.py --epoch=100
```

Manually resume transfer learning
```
python3 transfer_learning.py --resume --epoch=100 
```

Manually resume transfer learning with batch size 4
```
python3 transfer_learning.py --resume --epoch=100 --batch=4
```

Test specific image after model trained
```
python3 transfer_learning.py --resume --epoch=100 --image=glider.jpeg
```

Export trained model to ONNX format file - model.onnx
```
python3 transfer_learning.py --resume --epoch=100 --onnx
```

Export model weights to .wts file 
```
python3 transfer_learning.py --resume --epoch=100 --wts
```

Evalute trained model
```
python3 eval.py --image=paraglider.png
```

## Reference

https://learnopencv.com/image-classification-using-transfer-learning-in-pytorch/
https://github.com/spmallick/learnopencv/blob/master/Image-Classification-in-PyTorch/image_classification_using_transfer_learning_in_pytorch.ipynb

## TensorRT

Enable tensorrt environment 

```
cd ~/venv/
source tensorrt/bin/activate
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1
```

Generate file model.engine from model.onnx and speed up inference by TensorRT 

From https://github.com/keras-team/keras-tuner/issues/317
```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1
```

Load file model.onnx and export to model.engine
```
python3 trt_inference.py --onnx=model.onnx --image=glider.jpeg
```

Load file model.engine and speed up inference by TensorRT 
```
python3 trt_inference.py --engine=model.engine --image=glider.jpeg
```

Run inference by TensorRT C++ API
```
mkdir build
cd build/
cmake ../
./resnet50 -e ../model.engine -i ../glider.jpeg
```

Inference image folder
```
./resnet50 -e ../model.engine -d ../data/test/glider
```

## Reference

https://github.com/kuangliu/pytorch-cifar

https://github.com/spmallick/learnopencv/tree/master/PyTorch-ONNX-TensorRT
https://github.com/spmallick/learnopencv/tree/master/PyTorch-ONNX-TensorRT-CPP

https://github.com/pytorch/pytorch/issues/9176

https://github.com/NVIDIA/TensorRT/issues/183
https://github.com/NVIDIA/TensorRT/issues/730

https://github.com/onnx/onnx-tensorrt/blob/master/README.md

https://forumvolt.com/magic/how-to-install-sublime-text-3-on-jetson-nano.86/

https://github.com/onnx/tensorflow-onnx/issues/883

### 基于TensorRT C++ API 加速 TF 模型

https://blog.csdn.net/haiyangyunbao813/article/details/110209039

https://github.com/wang-xinyu/tensorrtx
https://github.com/wang-xinyu/pytorchx
https://github.com/BlueMirrors/torchtrtz
https://github.com/wdhao/tensorRT_Wheels

### Jetson nano

wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl \

PATH="/usr/local/cuda/bin:${PATH}" \
LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \

pip3 install pycuda six --verbose \

sudo apt-get install libopenblas-base libopenmpi-dev \
pip3 install matplotlib \
pip3 install albumentations==0.5.2 \
pip3 install torchsummary \

### Softmax v.s. LogSoftmax

https://zhenglungwu.medium.com/softmax-v-s-logsoftmax-7ce2323d32d3

### Normal Distribution

https://www.ycc.idv.tw/deep-dl_1.html

### Jetson Modules 

https://developer.nvidia.com/embedded/jetson-modules