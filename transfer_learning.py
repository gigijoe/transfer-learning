import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torchsummary import summary
import torch.backends.cudnn as cudnn

import onnx

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import struct

from PIL import Image

import sys
sys.path.append('./pytorch-cifar')

from models import * # This is from ../modules folder

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Transfer Training')
#parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from model')
parser.add_argument('--epoch', '-e', default=25, type=int, help='Epoch')
parser.add_argument('--batch', '-b', default=1, type=int, help='Batch size')
parser.add_argument("--image", '-i',type=str,
                        help="image file to test trained model")
parser.add_argument('--onnx', '-o', action='store_true',
                    help='Export to ONNX')
parser.add_argument('--wts', '-w', action='store_true',
                    help='Export to .wts weights')
args = parser.parse_args()

# Applying Transforms to the Data
image_transforms = { 
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=34, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=32),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ]),
    'valid': transforms.Compose([
        #transforms.RandomResizedCrop(size=36, scale=(0.8, 1.0)),
        #transforms.RandomRotation(degrees=15),
        #transforms.RandomHorizontalFlip(),
        #transforms.CenterCrop(size=32),
        transforms.Resize(size=34),
        transforms.CenterCrop(size=32),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=34),
        transforms.CenterCrop(size=32),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ])
}

dataset = 'data'

train_dir = os.path.join(dataset, 'train')
valid_dir = os.path.join(dataset, 'valid')
test_dir = os.path.join(dataset, 'test')

num_classes = len(os.listdir(train_dir))
print(f'number of classes is {num_classes}')

data = {
	'train' : datasets.ImageFolder(root=train_dir, transform=image_transforms['train']),
	'valid' : datasets.ImageFolder(root=valid_dir, transform=image_transforms['valid']),
	'test' : datasets.ImageFolder(root=test_dir, transform=image_transforms['test'])
}

# Get a mapping of the indices to the class names, in order to see the output classes of the test images.
idx_to_class = {v: k for k, v in data['train'].class_to_idx.items()}
print(idx_to_class)

# Size of Data, to be used for calculating Average Loss and Accuracy
train_data_size = len(data['train'])
valid_data_size = len(data['valid'])
test_data_size = len(data['test'])

print(train_data_size, valid_data_size, test_data_size)

bs = args.batch # Batch size

# Create iterators for the Data loaded using DataLoader module
train_data_loader = DataLoader(data['train'], batch_size=bs, shuffle=True)
valid_data_loader = DataLoader(data['valid'], batch_size=bs, shuffle=True)
test_data_loader = DataLoader(data['test'], batch_size=bs, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

resume_acc = 0.0
resume_loss = 100000.0
resume_epoch = 0

if args.resume:
    print(f'==> Load {dataset}.pt')
    resnet50 = torch.load(dataset+'.pt')
    print(resnet50)
    resnet50 = resnet50.to(device)
    if device == 'cuda':
        resnet50 = torch.nn.DataParallel(resnet50)
        cudnn.benchmark = True

    print('==> Load ./checkpoint/ckpt.pth')
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    resume_acc = checkpoint['acc']
    print(f'==> Resume accuracy : {resume_acc}')
    resume_loss = checkpoint['loss']
    print(f'==> Resume loss : {resume_loss}')
    resume_epoch = checkpoint['epoch']
    print(f'==> Resume epoch : {resume_epoch}')

else:
    print('==> Create ResNet50')
    resnet50 = ResNet50()
    #print(resnet50)
    resnet50 = resnet50.to(device)
    if device == 'cuda':
        resnet50 = torch.nn.DataParallel(resnet50)
        cudnn.benchmark = True

    print('==> Load ./pytorch-cifar/checkpoint/ckpt.pth')
    checkpoint = torch.load('./pytorch-cifar/checkpoint/ckpt.pth')
    print('==> load_state_dict')
    resnet50.load_state_dict(checkpoint['net'])

    # Freeze model parameters
    for param in resnet50.parameters():
        param.requires_grad = False

    # Change the final layer of ResNet50 Model for Transfer Learning
    fc_inputs = resnet50.module.linear.in_features
    '''
    resnet50.module.linear = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes), # Since 10 possible outputs
        #nn.LogSoftmax(dim=1) # For using NLLLoss()
        nn.Softmax(dim=1) # For using CrossEntropyLoss()
    )
    '''
    '''
    resnet50.module.linear = nn.Linear(fc_inputs, num_classes)
    '''
    
    resnet50.module.linear = nn.Sequential(
        nn.Linear(fc_inputs, num_classes), # Since 10 possible outputs
        nn.LogSoftmax(dim=1) # For using NLLLoss()
        #nn.Softmax(dim=1) # For using CrossEntropyLoss()
    )
    
    # Convert model to be used on GPU
    resnet50 = resnet50.to(device)

    print(resnet50)
# Define Optimizer and Loss Function
loss_func = nn.NLLLoss() # LogSoftmax
#loss_func = nn.CrossEntropyLoss() # Softmax
optimizer = optim.Adam(resnet50.parameters()) # Dynamic learning rate 
#optimizer = optim.SGD(resnet50.parameters(), lr=0.01, momentum=0.9)

def train_and_validate(model, loss_criterion, optimizer, epochs=25):
    '''
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param epochs: Number of epochs (default=25)
  
    Returns
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    '''
    
    start = time.time()
    history = []
    best_acc = resume_acc
    best_loss = resume_loss
    best_epoch = None

    for epoch in range(resume_epoch, epochs + resume_epoch):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs + resume_epoch))
        
        # Set to training mode
        model.train()
        
        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0
        
        valid_loss = 0.0
        valid_acc = 0.0
        
        for i, (inputs, labels) in enumerate(train_data_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Clean existing gradients
            optimizer.zero_grad()
            
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)
            
            # Compute loss
            loss = loss_criterion(outputs, labels)
            
            # Backpropagate the gradients
            loss.backward()
            
            # Update the parameters
            optimizer.step()
            
            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)
            
            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            
            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            
            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)
            
            #print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

        
        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            model.eval()

            # Validation loop
            for j, (inputs, labels) in enumerate(valid_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = loss_criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)

                #print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
        
        # Find average training loss and training accuracy
        avg_train_loss = train_loss/train_data_size 
        avg_train_acc = train_acc/train_data_size

        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss/valid_data_size 
        avg_valid_acc = valid_acc/valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
                
        epoch_end = time.time()
    
        print("Epoch : {:03d}, Training: Loss - {:.4f}, Accuracy - {:.4f}%, \n\t\tValidation : Loss - {:.4f}, Accuracy - {:.4f}%, Time: {:.4f}s".format(epoch, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
        
        # Save if the model has best accuracy till now
        #torch.save(model, dataset+'_model_'+str(epoch)+'.pt')
        #if valid_acc > best_acc:
        #if avg_valid_loss < best_loss:
        if avg_valid_loss < best_loss and avg_valid_acc >= best_acc:
            best_acc = avg_valid_acc
            best_loss = avg_valid_loss
            best_epoch = epoch
            print(f'==> Best accuracy : {best_acc}')
            print(f'==> Best loss : {best_loss}')
            print(f'Save {dataset}.pt')
            torch.save(model.module, dataset+'.pt') # Strip out module from DataParallel
            
            state = {
                'acc': best_acc,
                'loss': best_loss,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')

    return model, history, best_epoch

# Train the model for 25 epochs
num_epochs = args.epoch
trained_model, history, best_epoch = train_and_validate(resnet50, loss_func, optimizer, num_epochs)

#torch.save(history, dataset+'_history.pt')

history = np.array(history)
plt.plot(history[:,0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(-1,1)
plt.savefig(dataset+'_loss_curve.png')
plt.show()

plt.plot(history[:,2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0,1)
plt.savefig(dataset+'_accuracy_curve.png')
plt.show()

def generate_weights(model, trt_weights_path: str) -> None:
    """Convert torch weights format to wts weights format
    Args:
        trt_weights_path (str): Path where trt weights will be saved.
    """
    # open tensorrt weights file
    wts_file = open(trt_weights_path, "w")

    # write length of keys
    print("Keys: ", model.state_dict().keys())
    wts_file.write("{}\n".format(len(model.state_dict().keys())))
    for key, val in model.state_dict().items():
        print("Key: {}, Val: {}".format(key, val.shape))
        vval = val.reshape(-1).cpu().numpy()
        wts_file.write("{} {}".format(key, len(vval)))
        for v_l in vval:
            wts_file.write(" ")

            # struct.pack Returns a bytes object containing the values v1, v2, …
            # packed according to the format string format (>big endian in this case).
            wts_file.write(struct.pack(">f", float(v_l)).hex())
        wts_file.write("\n")

    wts_file.close()

def computeTestSetAccuracy(model, loss_criterion):
    '''
    Function to compute the accuracy on the test set
    Parameters
        :param model: Model to test
        :param loss_criterion: Loss Criterion to minimize
    '''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_acc = 0.0
    test_loss = 0.0

    # Validation - No gradient tracking needed
    with torch.no_grad():

        # Set to evaluation mode
        model.eval()

        # Validation loop
        for j, (inputs, labels) in enumerate(test_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)

            # Compute loss
            loss = loss_criterion(outputs, labels)

            # Compute the total loss for the batch and add it to valid_loss
            test_loss += loss.item() * inputs.size(0)

            # Calculate validation accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to valid_acc
            test_acc += acc.item() * inputs.size(0)

            print("Test Batch number: {:03d}, Test: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

    # Find average test loss and test accuracy
    avg_test_loss = test_loss/test_data_size 
    avg_test_acc = test_acc/test_data_size

    print("Test accuracy : " + str(avg_test_acc))

def predict(model, test_image_name):
    '''
    Function to predict the class of a single test image
    Parameters
        :param model: Model to test
        :param test_image_name: Test image

    '''
    
    transform = image_transforms['test']

    test_image = Image.open(test_image_name).convert("RGB")
    plt.imshow(test_image)
    plt.show()

    test_image_tensor = transform(test_image)
    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 32, 32).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 32, 32)
    
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        output = model(test_image_tensor)
        #print(output)
        print(f"Output size : {output.size()}")
        output = output.squeeze()
        print(f"Output size after squeezing : {output.size()}")
        print(output)

        # Result postpro
        _, indices = torch.sort(output, descending=True)

        print("\n\nInference results:")
        for index in range(num_classes):
            print(f"Label {index}: {idx_to_class[index]} ({output[index].item():.4f})")

        _, predicted = torch.max(output, 0)
        print(f'Best : {idx_to_class[predicted.item()]}')

    return test_image_tensor


#model = torch.load("{}_model_{}.pt".format(dataset, best_epoch))
print(f'==> Load {dataset}.pt')
model = torch.load(dataset+'.pt')
print(model)
# Load Data from folders
computeTestSetAccuracy(model, loss_func)

# Test a particular model on a test image
if args.image:
    input = predict(model, args.image)

if args.onnx:
    print('Export model to ONNX format ...')
    input = torch.randn(1, 3, 32, 32, device='cuda')
    # convert to ONNX --------------------------------------------------------------------------------------------------
    ONNX_FILE_PATH = "model.onnx"
    torch.onnx.export(model, input, ONNX_FILE_PATH, input_names=["input"], output_names=["output"], export_params=True, do_constant_folding=True, verbose=True)

    onnx_model = onnx.load(ONNX_FILE_PATH)
    # check that the model converted fine
    onnx.checker.check_model(onnx_model)

    print("Model was successfully converted to ONNX format.")
    print("It was saved to", ONNX_FILE_PATH)

if args.wts:
    print('Export model weights to .wts file ...')
    generate_weights(model, dataset+'.wts')
    print("Model weights was successfully converted to .wts file.")
    print(f'It was saved to {dataset}.wts')
