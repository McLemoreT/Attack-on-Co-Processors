import torch
import torchvision
from torch.autograd import Variable
import memtorch
from memtorch.mn.Module import patch_model
from memtorch.map.Input import naive_scale
from memtorch.map.Parameter import naive_map
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from memtorch.utils import LoadMNIST
import numpy as np
import torchvision.transforms as transforms
import copy
from os.path import exists # Added this to test if there is already a trained network
from PIL import Image
import matplotlib.pyplot as plt
import argparse # For parsing arguments from command line
import pandas as pd
import time 
import _thread

from deepfool import deepfool
from patch import patchIdeals

parser = argparse.ArgumentParser() #Create parser variable for command line arguments
parser.add_argument("-l", "--load_model", help="Disables automatically loading and useing a trained model if found", action="store_true")
parser.add_argument("-v", "--verbose", help="Show all additional information", action="store_true")
args = parser.parse_args()

torch.manual_seed(0) #seeds the array for consistent results

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def test(model, test_loader):
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data.to(device))
        pred = output.data.max(1)[1]
        correct += pred.eq(target.to(device).data.view_as(pred)).cpu().sum()

    return 100. * float(correct) / float(len(test_loader.dataset))

def getFoolData(model, test_loader):
    numArr = np.zeros((10, 10)) # Empty 10x10 array set up for: columns for original images 0 - 9, and corresponding columns for perturbed images 0 - 9. TODO: Make this automatically adjust instead of 10x10
    count = 0 # Represents count of loops, or the image that it's currently on
    datasetSize = len(fool_set) # Length of dataset
    filename = str(fool_set).partition('\n')[0].replace('Dataset', '').strip() + '_' + time.strftime("%m-%d-%Y_%H.%M.%S") + '.csv' # File saved is "Dataset Name_Date_Time"
    print('Storing Results in \"' + filename + '\"')
    df = pd.DataFrame(numArr) # Initializes the array
    
    print('Iterating through dataset of size', datasetSize)
    print('Starting...')
    for batch in fool_loader: # Loads all batches in loader
        for image in batch[0]: # Loads all images in current batch
            r, loop_i, label_orig, label_pert, pert_image = deepfool(image, model) # Run deepfool on the current image and model. r - perturbation vector
            numArr[label_pert, label_orig] += 1 # Rows = Perturbed images, Columns = Original Images. Adds 1 to wherever the fool of the actual value at the perturbed value took place
            count += 1 
            if(count % 50 == 0): # prints and logs progress every 50 counts
                print(count * 100 / datasetSize,'%') # progress as a percentage
                df.to_csv(filename, index=False) # Export and overwrite results to the CSV
            
    print('Displaying Results: Column = Original Image, Row = Matched Perturbed Image')
    print(numArr) 
    print('Successfully ran through', count, 'of', datasetSize, 'images.\nAll results stored in ', filename)
    return r, loop_i, label_orig, label_pert, pert_image; # Not actually needed but might as well keep just in case

def getFoolDataMultiThread(model, test_loader):
    numArr = np.zeros((10, 10)) # Empty 10x10 array set up for: columns for original images 0 - 9, and corresponding columns for perturbed images 0 - 9. TODO: Make this automatically adjust instead of 10x10
    count = 0 # Represents count of loops, or the image that it's currently on
    datasetSize = len(fool_set) # Length of dataset
    filename = str(fool_set).partition('\n')[0].replace('Dataset', '').strip() + '_' + time.strftime("%m-%d-%Y_%H.%M.%S") + '.csv' # File saved is "Dataset Name_Date_Time"
    print('Storing Results in \"' + filename + '\"')
    df = pd.DataFrame(numArr) # Initializes the array
    
    print('Iterating through dataset of size', datasetSize)
    print('Starting...')
    for batch in fool_loader: # Loads all batches in loader
        for image in batch[0]: # Loads all images in current batch
            r, loop_i, label_orig, label_pert, pert_image = deepfool(image, model) # Run deepfool on the current image and model. r - perturbation vector
            numArr[label_pert, label_orig] += 1 # Adds 1 to wherever the fool of the actual value at the perturbed value took place
            count += 1 
            if(count % 50 == 0): # prints and logs progress every 50 counts
                print(count * 100 / datasetSize,'%') # progress as a percentage
                df.to_csv(filename, index=False) # Export and overwrite results to the CSV
            
    print('Displaying Results: Column = Original Image, Row = Matched Perturbed Image')
    print(numArr) 
    print('Successfully ran through', count, 'of', datasetSize, 'images.\nAll results stored in ', filename)
    return r, loop_i, label_orig, label_pert, pert_image; # Not actually needed but might as well keep just in case


if __name__ == '__main__':

    device = torch.device('cpu' if 'cpu' in memtorch.__version__ else 'cuda')
    epochs = 10
    learning_rate = 1e-1
    step_lr = 5
    batch_size = 256
    train_loader, validation_loader, test_loader = LoadMNIST(batch_size=batch_size, validation=False)
    model = Net().to(device) # model created, some random thing
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_accuracy = 0

    train_network = True
    #Stuff I changed



    if exists('trained_model.pt'): # If model exists
        #model = TheModelClass(*args, **kwargs)
        model.load_state_dict(torch.load('trained_model.pt')) # Load it
        print('Found and loaded existing model:')
        print(model.eval())
        accuracy = test(model, test_loader)
        print('Model accuracy : %2.2f%%' % accuracy)
        if args.load_model: # checks for -L argument, automatic yes if true
            print(args.load_model)
            print('Do you want to use this model?')
            response = input("Type 'yes' or 'no':")
            response = response.lower()
        else:
            response = 'yes' #Automatically assumes you want to use the existing model if no flag is passed

            
        if response == 'yes': #Checks if user wants to use existing model
            train_network = False
            print('Using loaded model')
        else:
            print('training new model')
            best_accuracy = accuracy
    else:
        print('Model not found. Training new model.')
        accuracy = test(model, test_loader) # Will be very low, but needed for comparison improvement
        train_network = True #starts training




    #End of stuff I changed

    if train_network: # general pytorch code for training network

        for epoch in range(0, epochs):
            print('Epoch: [%d]\t\t' % (epoch + 1), end='')
            if epoch % step_lr == 0:
                learning_rate = learning_rate * 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate

            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data.to(device))
                loss = criterion(output, target.to(device))
                loss.backward()
                optimizer.step()

            accuracy = test(model, test_loader)
            print('%2.2f%%' % accuracy)
            if accuracy > best_accuracy:
                torch.save(model.state_dict(), 'trained_model.pt')
                best_accuracy = accuracy


    model = Net().to(device)
    model.load_state_dict(torch.load('trained_model.pt'), strict=False)



    transform = transforms.Compose([transforms.ToTensor()])
    fool_set = torchvision.datasets.MNIST(
        root="data", train=False, transform=transform, download=True
    )
    fool_loader = torch.utils.data.DataLoader(
        fool_set, batch_size=1, shuffle=False, num_workers=2
    )
    example = next(iter(fool_loader))[0][0] #TODO: This may not be correct

    r, loop_i, label_orig, label_pert, pert_image = deepfool(example , model) # Run a single test
#    print("Original label = ", label_orig)
#    print("Perturbed label = ", label_pert)
#    print("Perturbation Vector = ", np.linalg.norm(r))
    
    patch = False

    if(patch):
        patchedModel = patchIdeals(model)
    else:
        patchedModel = model
        
    r, loop_i, label_orig, label_pert, pert_image = getFoolDataMultiThread(patchedModel, test_loader) # Runs the entire dataset   

    def clip_tensor(A, minv, maxv):
        A = torch.max(A, minv*torch.ones(A.shape))
        A = torch.min(A, maxv*torch.ones(A.shape))
        return A
    

    
    clip = lambda x: clip_tensor(x, 0, 1)
    
    #These are uneccessary because this set is "grayscale"
    #std = [ 0.229, 0.224, 0.225 ]
    #mean = [ 0.485, 0.456, 0.406 ]
    
    tf = transforms.Compose([transforms.Normalize((0.5,), (0.5,)), #because grayscale
        transforms.Lambda(clip),
        transforms.ToPILImage(),
        transforms.CenterCrop(28)])
        
    #Display perturbed image
    plt.figure()
    plt.imshow(tf(pert_image.cpu()[0])) #shows it
    plt.title(label_pert) 
    plt.savefig("Image_Fooled.png") #saves to disk
    plt.show()
    
    #Display original image
    original_image = np.array(example, dtype='float')
    pixels = original_image.reshape((28, 28))
    plt.imshow(pixels) 
    plt.savefig("Image_Original.png")
    plt.show()
