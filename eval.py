'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms

import os
import argparse

import sys
sys.path.append('./pytorch-cifar')

from models import * # This is from ../modules folder
from utils import progress_bar

from PIL import Image
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Transfer Learning Evaluate')
parser.add_argument("--image", '-i',type=str,
						default="image.jpeg", help="image file")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = 'data'

train_dir = os.path.join(dataset, 'train')
num_classes = len(os.listdir(train_dir))
print(f'number of classes is {num_classes}')
# Get a mapping of the indices to the class names, in order to see the output classes of the test images.
idx_to_class = {v: k for k, v in datasets.ImageFolder(root=train_dir).class_to_idx.items()}
print(idx_to_class)

print(f'==> Load {dataset}.pt')
net = torch.load(dataset+'.pt')
print(net)
net = net.to(device)
net.eval()

def preprocess_image(img_path):
	# transformations for the input data
	transform_test = transforms.Compose([
		transforms.Resize((32, 32)),
		transforms.ToTensor(),
		transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
	])

	# read input image
	input_img = Image.open(img_path).convert("RGB")
	
	# do transformations
	input_data = transform_test(input_img)
	print(f'Input shape : {input_data.shape}')
	# prepare batch
	batch_data = torch.unsqueeze(input_data, 0)
	print(f'Input shape after unsqueeze : {batch_data.shape}')

	plt.figure()
	plt.imshow(input_img)
	plt.show()

	return batch_data

input = preprocess_image(args.image).cuda()
with torch.no_grad():
	input = input.to(device)
	output = net(input)
	#print(output)
	print(f"Output size : {output.size()}")
	output = output.squeeze()
	print(f"Output size after squeezing : {output.size()}")
	print(output)

	# Result postpro
	_, indices = torch.sort(output, descending=True)

	print("\n\nInference results:")
	for index in indices[:num_classes]:
		print(f"Label {index}: {idx_to_class[index.item()]} ({output[index].item():.4f})")

	_, predicted = torch.max(output, 0)
	print(f'Best : {idx_to_class[predicted.item()]}')
