import torch
import memtorch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
from os.path import exists # Added this to test if there is already a trained network
import matplotlib.pyplot as plt
import argparse # For parsing arguments from command line
import pandas as pd
import time 


from MNISTMethods import MNISTMethods
from CIFAR10Methods import CIFAR10Methods
from deepfool import deepfool
from patch import patchIdeals

#Our other files
import tests as tests

parser = argparse.ArgumentParser() #Create parser variable for command line arguments
# Just use "run testing.py [arguments]" to run in spyder console
# May need to cd to correct directory

parser.add_argument("-l", "--load_model", help="Disables automatically loading and useing a trained model if found", action="store_true")
parser.add_argument("-v", "--verbose", help="Show all additional information", action="store_true")
parser.add_argument("-d", "--demo", help="Runs a single test instead of the full set of images", action="store_true")

# Non-Ideality Processing
parser.add_argument("-D", "--nonID_DeviceFaults", help="Applies DeviceFaults nonideality and prints the results.", action="store_true")
parser.add_argument("-E", "--nonID_Endurance", help="Applies Endurance non-ideality and prints the results.", action="store_true")
parser.add_argument("-R", "--nonID_Retention", help="Applies Retention non-ideality and prints the results.", action="store_true")
parser.add_argument("-F", "--nonID_FiniteConductanceStates", help="Applies FiniteConductanceStates non-ideality and prints the results.", action="store_true")
parser.add_argument("-N", "--nonID_NonLinear", help="Applies NonLinear non-ideality and prints the results.", action="store_true")
parser.add_argument("-MNIST", "--MNIST", help="Uses the MNIST Dataset and models", action="store_true")
parser.add_argument("-CIFAR10", "--CIFAR10", help="Uses the CIFAR10 Dataset and models", action="store_true")

# Test functiosns in tests.oy
parser.add_argument("-TEST-GOOD-PERTURB", "--TEST_GOOD_PERTURB", help="Runs the Good Perturb test", action="store_true")
parser.add_argument("-TEST-QUARRY", "--TEST_QUARRY", help="Runs the Quarry test", action="store_true")
parser.add_argument("-TEST-BORE", "--TEST_BORE", help="Runs the Quarry test", action="store_true")
args = parser.parse_args()

torch.manual_seed(0) #seeds the array for consistent results
# Comment out for random images + perturbations


def test(model, test_loader):
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data.to(device))
        pred = output.data.max(1)[1]
        correct += pred.eq(target.to(device).data.view_as(pred)).cpu().sum()

    return 100. * float(correct) / float(len(test_loader.dataset))

def getFoolData(model, test_loader):
    # Figuring out size of output layer
    params = list(model.parameters())
    outputSize = int(str(params[len(params) - 1].size()).strip().replace('torch.Size([', '').replace('])', ''))
    print("Output Size for this network: ", outputSize)
    numArr = np.zeros((outputSize, outputSize)) # Empty array set to be the size of the final output layer. Compares difference between result and input
    # For example, MNIST will initialize an empty 10x10 array set up for: columns for original images 0 - 9, and corresponding rows for perturbed images 0 - 9.
    
    count = 0 # Represents count of loops, or the image that it's currently on
    datasetSize = len(fool_set) # Length of dataset
    
    nonIDs = ""
    if (args.nonID_DeviceFaults):
        nonIDs += "DeviceFaults-"
    if (args.nonID_Endurance):
        nonIDs += "Endurance-"
    if (args.nonID_Retention):
        nonIDs += "Retention-"
    if (args.nonID_FiniteConductanceStates):
        nonIDs += "FiniteConductanceStates-"
    if (args.nonID_NonLinear):
        nonIDs += "NonLinear-" 
    nonIDs = nonIDs[:-1] # Removes the last '-'
    filename = str(fool_set).partition('\n')[0].replace('Dataset', '').strip() + '_'  + nonIDs + '_' + time.strftime("%m-%d-%Y_%H.%M.%S") + '.csv' # File saved is "Dataset Name_Date_Time"
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



    
  
if __name__ == '__main__':
    
    start_time = time.time()
    
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

    if exists(modelName): # If model exists
        #model = TheModelClass(*args, **kwargs)
        
        #        print('Found and loaded existing model:')
        #        print(model.eval())
        #        accuracy = test(model, test_loader)
        #        print('Model accuracy : %2.2f%%' % accuracy)
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
            model.load_state_dict(torch.load(modelName)) # Load it
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
        fool_set, batch_size=100, shuffle=False, num_workers=2
    ) # For random images + perturbations, turn shuffle = True


    patch = True # If true, pay attention to non-ideality flags and apply them

    if(patch):
        patchedModel = patchIdeals(model, args) # The non-idealities get applied to the model
        print(args)
        torch.save(patchedModel.state_dict(), 'patched_trained_model' + '.pt')
    else:
        patchedModel = model # The model stays as itself
        
        
    if (args.demo): # If true, a single image will be tested. If false, the full image database will be tested (This could take hours).
        example = next(iter(fool_loader))[0][0] #TODO: This may not be correct
        r, loop_i, label_orig, label_pert, pert_image = deepfool(example, patchedModel) # Run a single test        
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

        
    #Display perturbed image
    if args.verbose:
            
        tf = transforms.Compose([transforms.Normalize((0.5,), (0.5,)), #because grayscale
        transforms.Lambda(clip),
        transforms.ToPILImage(),
        transforms.CenterCrop(32)])
        
        plt.figure()
        plt.ion()
        plt.imshow(tf(pert_image.cpu()[0])) #shows it
        plt.suptitle("Fooled (Perturbed) Image")
        plt.title("Classification: " + str(label_pert)) # It's supposed to be suptitle not subtitle
        plt.savefig("Image_Fooled.png") #saves to disk
        plt.show()
        plt.close()

    end_time = time.time()
    print("Time taken: ", end_time - start_time)

    #Add command line argument check here
    #if(args.TEST_QUARRY):
    #    tests.isGoodPlaceTest(polyset, model, patchedModel, fool_set)
    if(args.TEST_QUARRY):
        tests.bore(model, patchedModel, fool_set)
    if(args.TEST_GOOD_PERTURB):
        tests.goodPerturbTest(fool_set, model, patchedModel)
    if(args.TEST_BORE):
        tests.bore(fool_set, model, patchedModel)
        
