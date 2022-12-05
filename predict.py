# PROGRAMMER: Lama AlDaej
# DATE CREATED: 20 March, 2022                              
# PURPOSE: Use a trained network to predict the class for an input image. 
#          Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.
# ------------------------------------------------------------------------------------------------------------------

# Import essential libraries
import torch
from os.path import isdir
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
from torch import nn
from torch import optim
from collections import OrderedDict
import PIL
from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Basic usage: python predict.py /path/to/image checkpoint
# ArgumentParser parses arguments through the parse_args () method. 
# This will inspect the command line, convert each argument to the appropriate type and then invoke the appropriate action.
# References: 
#   - https://docs.python.org/3/library/argparse.html#module-argparse
#   - https://stackoverflow.com/questions/51823427/python-how-to-accept-arguments-via-command-line-using-argumentparser
#   - https://www.educba.com/python-argparse/
def parse_arguments():
    # An argument parser object is created named “parser”
    parser = argparse.ArgumentParser(description = 'Predict Arguments')
    
    # Get image path
    parser.add_argument('--img', type=str, default = 'flowers/test/10/image_07090.jpg')
    
    # Get checkpoint path
    parser.add_argument('--checkpoint', type=str, default = 'model_checkpoint.pth')
    
    # Option 1: Return top K most likely classes 
    #   python predict.py input checkpoint --top_k 3
    parser.add_argument('--top_k', type=int, default = 5)
    
    # Option 2: Use a mapping of categories to real names
    #   python predict.py input checkpoint --category_names cat_to_name.json
    parser.add_argument('--category_names', type=str, default = 'cat_to_name.json')
    
    # Option 3: Use GPU for inference
    #   python predict.py input checkpoint --gpu
    # store_true action is utilized to take care of the boolean switches when it comes to using the argparse() functionality.
    parser.add_argument('--gpu', action="store_true")
    
    args = parser.parse_args()
    
    return args

# Load and rebuild the model
# Reference:  https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
def load_checkpoint_rebuild_model(path):
    checkpoint = torch.load(path)
    architecture = checkpoint['architecture']
    model = models.vgg16(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


# Preprocess the image
# References: 
#   - https://www.projectpro.io/recipes/crop-and-resize-image-pytorch
#   - https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
#   - https://stackoverflow.com/questions/4321290/how-do-i-make-pil-take-into-account-the-shortest-side-when-creating-a-thumbnail
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    
    pil_image = Image.open(image)

    # Resize the images where the shortest side is 256 pixels, keeping the aspect ratio. 
    
    # Get the width and height
    original_width = pil_image.size[0]
    original_height = pil_image.size[1]
    resize = [0,0]
    
    # Set the shorter aspect as 256
    if original_width < original_height:
        resize = [256, pil_image.size[1]]
    else:
        resize = [pil_image.size[0], 256]
    
    pil_image.thumbnail(size=resize)
    
    # Crop out the center 224x224 portion of the image.
    left = (256 - 224)/2
    top = (256 - 224)/2
    right = (256 + 224)/2
    bottom = (256 + 224)/2

    pil_image = pil_image.crop((left, top, right, bottom))
    

    # Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. 
    # So, convert the values 
    np_image = np.array(pil_image)/255
    
    # The network expects the images to be normalized in a specific way. 
    # For the means, it's [0.485, 0.456, 0.406] and for the standard deviations [0.229, 0.224, 0.225]. 
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Subtract the means from each color channel, then divide by the standard deviation.           
    np_image = ((np_image - mean) / std)
    
    # Reverse or permute the axes of an array; returns the modified array.
    # Reference: https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
    np_image = np.transpose(np_image, (2, 0, 1))
    
    return np_image

# Predict the top k classes of an image
def predict(image_path, model, topk, category_names,args):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    
    model.to('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    model.eval()
    
    # Process a PIL image for use in a PyTorch model
    img = process_image(image_path)
    # Set the image numpy array as a torch object
    img = torch.from_numpy(np.expand_dims(img,axis=0)).type(torch.FloatTensor).to('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

    # calculate the class probabilities
    prob = model.forward(img)

    # Scale the probabilities from log to linear
    scale_prob = torch.exp(prob)
    
    # Get the top k classes and probabilities
    top_prob, top_class= scale_prob.topk(topk)

    # Set the image torch object as numpy array
    # Reference: https://sparrow.dev/pytorch-numpy-conversion/
    top_prob = np.array(top_prob.detach())[0]
    top_class = np.array(top_class.detach())[0]
    
    with open(category_names, 'r') as json_file:
        cat_to_name = json.load(json_file)
    
    # Convert the indices to the actual class labels using class_to_idx 
    idx_to_class = {clss:cat_to_name[idx] for idx, clss in model.class_to_idx.items()}  
    # Invert the dictionary so you get a mapping from index to class as well.
    top_class = [idx_to_class[idx] for idx in top_class]

    return top_prob, top_class

# Print the most likely image class and it's associated probability
def print_class_probability(top_prob, top_class):
    for i in range(len(top_class)):
        print(f'Class {i+1}: {top_class[i]} with probability = {top_prob[i]}')
    return None
    

def main():
    # Get the arguments
    args = parse_arguments()
    
    # Load the model
    model = load_checkpoint_rebuild_model(args.checkpoint)

    # Predict the top k classes of the image
    top_prob, top_class = predict(args.img, model, args.top_k, args.category_names, args)
    
    # Print the most likely image class and it's associated probability
    print_class_probability(top_prob, top_class)
    
if __name__ == '__main__': 
    main()