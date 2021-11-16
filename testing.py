import torch
import torchvision
from torch.autograd import Variable
import memtorch
from memtorch.mn.Module import patch_model
from memtorch.map.Input import naive_scale
from memtorch.map.Parameter import naive_map
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
import copy
from os.path import exists # Added this to test if there is already a trained network
from PIL import Image
import matplotlib.pyplot as plt
import argparse # For parsing arguments from command line
import pandas as pd
import time 
import threading

from MNISTMethods import MNISTMethods
from CIFAR10Methods import CIFAR10Methods
from deepfool import deepfool
from patch import patchIdeals

parser = argparse.ArgumentParser() #Create parser variable for command line arguments
# Just use "run testing.py [arguments]" to run in python

parser.add_argument("-l", "--load_model", help="Disables automatically loading and useing a trained model if found", action="store_true")
parser.add_argument("-v", "--verbose", help="Show all additional information", action="store_true")

# Non-Ideality Processing
parser.add_argument("-D", "--nonID_DeviceFaults", help="Applies DeviceFaults nonideality and prints the results.", action="store_true")
parser.add_argument("-E", "--nonID_Endurance", help="Applies Endurance non-ideality and prints the results.", action="store_true")
parser.add_argument("-R", "--nonID_Retention", help="Applies Retention non-ideality and prints the results.", action="store_true")
parser.add_argument("-F", "--nonID_FiniteConductanceStates", help="Applies FiniteConductanceStates non-ideality and prints the results.", action="store_true")
parser.add_argument("-N", "--nonID_NonLinear", help="Applies NonLinear non-ideality and prints the results.", action="store_true")
parser.add_argument("-MNIST", "--MNIST", help="Uses the MNIST Dataset and models", action="store_true")
parser.add_argument("-CIFAR10", "--CIFAR10", help="Uses the CIFAR10 Dataset and models", action="store_true")

args = parser.parse_args()
usedArgs = parser.parse_known_args()

torch.manual_seed(0) #seeds the array for consistent results



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
    print('Total threads:', fool_loader.num_workers)
    print('Starting...')
    for batch in fool_loader: # Loads all batches in loader
        start_time = time.time() # Start timer
        for image in batch[0]: # Loads all images in current batch
            r, loop_i, label_orig, label_pert, pert_image = deepfool(image, model) # Run deepfool on the current image and model. r - perturbation vector
            numArr[label_pert, label_orig] += 1 # Rows = Perturbed images, Columns = Original Images. Adds 1 to wherever the fool of the actual value at the perturbed value took place
            count += 1 
            if(count % 50 == 0): # prints and logs progress every 50 counts
                print(count * 100 / datasetSize,'%') # progress as a percentage
                df.to_csv(filename, index=False) # Export and overwrite results to the CSV
        end_time = time.time() # End timer
        
    print('Displaying Results: Column = Original Image, Row = Matched Perturbed Image')
    print(numArr) 
    print('Successfully ran through', count, 'of', datasetSize, 'images.\nAll results stored in ', filename)
    print('Total Execution Time:', end_time - start_time  )
    return r, loop_i, label_orig, label_pert, pert_image; # Not actually needed but might as well keep just in case

def getFoolDataMultiThread(model, test_loader):
    
    threadAmt = 2
    thread1 = getFoolDataThread(1, "Thread-1", 1)
    thread2 = getFoolDataThread(2, "Thread-2", 2)
    print('Active thread count:', threading.activeCount())


#    print('Iterating through dataset of size', datasetSize)
#    print('Allocating', int(datasetSize / threadAmt), 'samples each for', threadAmt, 'active threads.')
    print('Starting...')
    thread1.start()
    thread2.start()
            
    thread1.join()
    thread2.join()
    print('Displaying Results: Column = Original Image, Row = Matched Perturbed Image')

#    print(numArr) 
#    print('Successfully ran through', count, 'of', datasetSize, 'images.\nAll results stored in ', filename)
    return r, loop_i, label_orig, label_pert, pert_image; # Not actually needed but might as well keep just in case

class getFoolDataThread(threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self._stop_event = threading.Event()
        
    def run(self): # Runs when .start() is called
        print ("Starting " + self.name)
        numArr = np.zeros((10, 10)) # Empty 10x10 array set up for: columns for original images 0 - 9, and corresponding columns for perturbed images 0 - 9. TODO: Make this automatically adjust instead of 10x10
        count = 0
        datasetSize = len(fool_set) # Length of dataset
        filename = str(fool_set).partition('\n')[0].replace('Dataset', '').strip() + '_' + + time.strftime("%m-%d-%Y_%H.%M.%S") + '.csv' # File saved is "Dataset Name_Date_Time"
        print('Storing Results in \"' + filename + '\"')
        df = pd.DataFrame(numArr) # Initializes the array
        print('BATCH AMOUNT:', fool_loader.batch_size)

        for batch in fool_loader: # Loads all batches in loader
            print('IMAGE AMOUNT:', len(batch))
            for image in batch[0]: # Loads all images in current batch
                r, loop_i, label_orig, label_pert, pert_image = deepfool(image, model) # Run deepfool on the current image and model. r - perturbation vector
                numArr[label_pert, label_orig] += 1 # Adds 1 to wherever the fool of the actual value at the perturbed value took place
                count += 1 
                if(count % 50 == 0): # prints and logs progress every 50 counts
                    print(count * 100 / datasetSize,'%') # progress as a percentage
                    df.to_csv(filename, index=False) # Export and overwrite results to the CSV
                    break
        print ("Exiting " + self.name)
        
    def stop(self):
        self._stop_event.set()
    
if __name__ == '__main__':
    if(args.MNIST): #selection of dataset and corresponding Net/methods
        polyset = MNISTMethods()
    elif(args.CIFAR10):
        polyset = CIFAR10Methods()
    else:
        polyset = MNISTMethods()
        
    device = torch.device('cpu' if 'cpu' in memtorch.__version__ else 'cuda')
    epochs = 10
    learning_rate = 1e-1
    step_lr = 5
    batch_size = 256
    
    
    fool_set, train_loader, validation_loader, test_loader = polyset.dataReturn(batch_size) #polymorphic set initialization
    model = polyset.returnNetToDevice(device) #polymorphic net shape call
    modelName = polyset.getName() #return model name
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_accuracy = 0

    train_network = True

#    print('Args: ', args)    
#    magic = str(args).partition('\n')[0].strip().replace('Namespace(', '')
#    str(args).partition('\n')[0].replace('=False,', '').replace('=True,', '')

#    print('known args: ', vars(args))
 #   filename = str(fool_set).partition('\n')[0].replace('Dataset', '').strip() + '_' + '_'.join(args) + '_' + time.strftime("%m-%d-%Y_%H.%M.%S") + '.csv' # File saved is "Dataset Name_Date_Time"
 #   print('Storing Results in \"' + filename + '\"')


    if exists(modelName): # If model exists
        #model = TheModelClass(*args, **kwargs)
        model.load_state_dict(torch.load(modelName)) # Load it
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
                torch.save(model.state_dict(), modelName)
                best_accuracy = accuracy


    model = polyset.returnNetToDevice(device) #polymorphic net shape call
    model.load_state_dict(torch.load(modelName), strict=False)

    fool_loader = torch.utils.data.DataLoader(
        fool_set, batch_size=100, shuffle=False, num_workers=8
    )
    example = next(iter(fool_loader))[0][0] #TODO: This may not be correct

    r, loop_i, label_orig, label_pert, pert_image = deepfool(example , model) # Run a single test
#    print("Original label = ", label_orig)
#    print("Perturbed label = ", label_pert)
#    print("Perturbation Vector = ", np.linalg.norm(r))
    

    patch = True # If true, pay attention to non-ideality flags and apply them

    if(patch):
        patchedModel = patchIdeals(model, args) # The non-idealities get applied to the model
    else:
        patchedModel = model # The model stays as itself
        
    compareNonIdealities = False # If true, separate tests will be run for each non-ideality flag, both indvidually and in groups. This will usually take several hours.
                                 # If false, a single test will be run with all flags. 
    
    if compareNonIdealities:
        # Do Stuff
        print("")
    else:
        r, loop_i, label_orig, label_pert, pert_image = getFoolData(patchedModel, test_loader) # Runs the entire dataset   
        
#    r, loop_i, label_orig, label_pert, pert_image = getFoolDataMultiThread(patchedModel, test_loader) # Runs the entire dataset   


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
    plt.suptitle("Fooled Image")
    plt.savefig("Image_Fooled.png") #saves to disk
    plt.show()
    
    #Display original image
    original_image = np.array(example, dtype='float')
    pixels = original_image.reshape((28, 28))
    plt.imshow(pixels) 
    plt.suptitle("Original Image")
    plt.savefig("Image_Original.png")
    plt.show()
