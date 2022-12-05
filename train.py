# PROGRAMMER: Lama AlDaej
# DATE CREATED: 20 March, 2022                              
# PURPOSE: Train a new network on a dataset and save the model as a checkpoint.
# ------------------------------------------------------------------------------------------------------------------

# Import essential libraries
import torch
from os.path import isdir
import torchvision
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
from torch import nn
from torch import optim
from collections import OrderedDict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Basic usage: python train.py data_directory
# ArgumentParser parses arguments through the parse_args () method. 
# This will inspect the command line, convert each argument to the appropriate type and then invoke the appropriate action.
# References: 
#   - https://docs.python.org/3/library/argparse.html#module-argparse
#   - https://stackoverflow.com/questions/51823427/python-how-to-accept-arguments-via-command-line-using-argumentparser
#   - https://www.educba.com/python-argparse/
def parse_arguments():
    # An argument parser object is created named “parser”
    parser = argparse.ArgumentParser(description = 'Network Arguments')
    
    # Get Data Directory
    parser.add_argument('--data_dir', type=str, default='flowers')
    
    # Option 1: Set directory to save checkpoints
    #   python train.py data_dir --save_dir save_directory
    parser.add_argument('--save_dir', type=str, default='model_checkpoint.pth')
    
    # Option 2: Choose architecture
    #   python train.py data_dir --arch "vgg13"
    parser.add_argument('--arch', type=str, default = 'vgg16')
    
    # Option 3: Set hyperparameters
    #   python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    parser.add_argument('--learning_rate', type=float, default = 0.001)
    parser.add_argument('--hidden_units', type=int, default = 4096)
    parser.add_argument('--epochs', type=int, default = 5)
    
    # Option 4: Use GPU for training
    #   python train.py data_dir --gpu
    # store_true action is utilized to take care of the boolean switches when it comes to using the argparse() functionality.
    parser.add_argument('--gpu', action="store_true")
    
    args = parser.parse_args()
    
    return args

# Define the transforms for the training, validation, and testing sets
def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define the transforms for the training, validation, and testing sets
    # For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects.

    # Apply transformations such as random scaling, cropping, and flipping to the training set.
    # Make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.

    # I added some data augmentations to increase the semantic coverage of a dataset. 
    # Also, it can improve the model without having to collect and label more data.
    # Reference: https://pytorch.org/vision/stable/transforms.html
    # Reference: https://blog.roboflow.com/why-and-how-to-implement-random-rotate-data-augmentation/
    training_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                             transforms.RandomRotation(90),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.RandomVerticalFlip(p=0.5),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], 
                                                                  [0.229, 0.224, 0.225])])
    # The validation and testing sets don't need any scaling or rotation transformations.
    # but the sets need to be resized then cropped to the appropriate size.
    testing_transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])
    validation_transform = transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], 
                                                                    [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    train_set = torchvision.datasets.ImageFolder(train_dir, transform=training_transform)
    test_set = torchvision.datasets.ImageFolder(test_dir, transform=testing_transform)
    valid_set = torchvision.datasets.ImageFolder(valid_dir, transform=validation_transform)


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    training_dataloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_dataloader =  torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)
    valid_dataloader =  torch.utils.data.DataLoader(valid_set, batch_size=32, shuffle=True)
    
    return training_dataloader, test_dataloader, valid_dataloader, train_set

# Load a pre-trained network 
# Reference: https://pytorch.org/vision/stable/models.html
def load_network(arch='vgg16'):
    #model = None
#     load_code = 'model = models.'+arch+'(pretrained=True)'
#     exec(load_code, globals(), globals())
    model = getattr(models, arch)(pretrained=True)
    #model = models.vgg16(pretrained=True)
    model.name = arch
    # Freeze parameters 
    # Reference: https://androidkt.com/pytorch-freeze-layer-fixed-feature-extractor-transfer-learning/
    for param in model.parameters():
        param.requires_grad = False
    return model

# Building the classifier
def build_classifier(model, hidden_units):
    
    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    classifier= nn.Sequential(
        OrderedDict([
            ('fc1', nn.Linear(in_features=25088, out_features=hidden_units, bias=True)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.3)),
            ('fc2', nn.Linear(in_features=hidden_units, out_features=102, bias=True)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

    model.classifier = classifier 

    return model.classifier

# Define the loss function and optimizer
def define_loss_func_optimizer(args, model, learning_rate):
    # Define the device as CUDA if gpu is available.
    # Reference: https://wandb.ai/wandb/common-ml-errors/reports/How-To-Use-GPU-with-PyTorch---VmlldzozMzAxMDk
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    # Move model to CUDA
    model = model.to(device)

    # Define the loss function
    criterion = nn.NLLLoss()
    criterion.to(device)

    # Define the optimizer
    # Optimizers require the parameters to optimize and a learning rate
    # Reference: https://programming.vip/docs/torch.optim-optimization-algorithm-optim-adam.html
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    return criterion, optimizer, device

# Train the classifier layers using backpropagation using the pre-trained network to get the features
def train_model(model, epochs, training_dataloader, valid_dataloader, device, optimizer, criterion):
    print('******* Training Started *******')
    # Define the number of steps
    steps = 0

    # Create lists to record training loss
    train_losses = []
    val_losses = []
    acc_list = []

    for e in range(epochs):
        training_loss = 0
        num_correct = 0 
        num_samples = 0

        for images, labels in training_dataloader:
            # Get data to gpu
            images = images.to(device)
            labels = labels.to(device)

            # Increase the steps by 1
            steps += 1

            # Set all the gradients to 0 for each batch
            optimizer.zero_grad()
            # Forward
            outputs = model.forward(images)
            # Compute the training loss
            loss = criterion(outputs, labels)

            # Backword
            # Compute the updated weights of all the model parameters
            loss.backward()

            # gradient descent
            optimizer.step()

            training_loss += loss.item()

        else:
            validation_loss = 0
            accuracy = 0

            with torch.no_grad():
                for images, labels in valid_dataloader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    validation_loss += criterion(outputs, labels)
                    _, predictions = outputs.max(1)
                    num_correct += (predictions == labels).sum().item()
                    num_samples += labels.size(0)
                accuracy = float(num_correct/num_samples * 100)
            train_losses.append(training_loss / len(training_dataloader))
            val_losses.append(validation_loss / len(valid_dataloader))
            acc_list.append(accuracy)
            print(f'In Epoch # {e+1} : Training Loss= {training_loss}, Validation Loss= {validation_loss}, Accuracy= {accuracy:.2f}%')
    print('******* Training Done *******')
    return model

# Do validation on the test set
def validate_model(model, test_dataloader, device):
    accuracy = 0
    num_correct = 0
    num_samples = 0
        
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predictions = outputs.max(1)
            num_correct += (predictions == labels).sum().item()
            num_samples += labels.size(0)
    accuracy = float(num_correct/num_samples * 100)
    print(f'The Accuracy of the Model is {accuracy:.2f}%')
    return None

def save_model(model,train_set, arch, epochs,optimizer, path):
    # Save the checkpoint 
    # Reference:  https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
    model.class_to_idx = train_set.class_to_idx

    torch.save({'architecture': arch,
                  'classifier': model.classifier,
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict()}, path)
    return None

def main():
    # Get the arguments
    args = parse_arguments()
    
    # Load the data
    training_dataloader, test_dataloader, valid_dataloader, train_set = load_data(args.data_dir)
    
    # Load a pre-trained network 
    model = load_network(arch=args.arch)
    
    # Build the classifier
    model.classifier = build_classifier(model,args.hidden_units)
    
    # Define the loss function and optimizer
    criterion, optimizer, device = define_loss_func_optimizer(args, model, args.learning_rate)
    
    # Train the model
    model = train_model(model, args.epochs, training_dataloader, valid_dataloader, device, optimizer, criterion)
    
    # Validate the model
    validate_model(model, test_dataloader, device)
    
    # Save the model
    save_model(model,train_set, args.arch, args.epochs,optimizer, args.save_dir)
    
if __name__ == '__main__': 
    main()