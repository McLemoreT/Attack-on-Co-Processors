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
from memtorch.bh.nonideality.NonIdeality import apply_nonidealities
import copy
from os.path import exists #Added this to test if there is already a trained network
from PIL import Image
import matplotlib.pyplot as plt
import argparse #For parsing arguments from command line

from deepfool import deepfool

parser = argparse.ArgumentParser() #Create parser variable for command line arguments
parser.add_argument("-l", "--load_model", help="Automatically loads and uses a trained model if found", action="store_true")
parser.add_argument("-v", "--verbose", help="Show all additional information", action="store_true")
args = parser.parse_args()
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

if __name__ == '__main__':
    reference_memristor = memtorch.bh.memristor.VTEAM
    reference_memristor_params = {'time_series_resolution': 1e-10}
    memristor = reference_memristor(**reference_memristor_params)
    if args.verbose:
        memristor.plot_hysteresis_loop()
        memristor.plot_bipolar_switching_behaviour()


    device = torch.device('cpu' if 'cpu' in memtorch.__version__ else 'cuda')
    epochs = 10
    learning_rate = 1e-1
    step_lr = 5
    batch_size = 256
    train_loader, validation_loader, test_loader = LoadMNIST(batch_size=batch_size, validation=False)
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_accuracy = 0

    train_network = True
    #Stuff I changed



    if exists('trained_model.pt'):
        #model = TheModelClass(*args, **kwargs)
        model.load_state_dict(torch.load('trained_model.pt'))
        print('Found and loaded existing model')
        print(model.eval())
        accuracy = test(model, test_loader)
        print('Model accuracy : %2.2f%%' % accuracy)

    if args.load_model:
        response = 'yes'
    else:
        print('Do you want to use this model?')
        response = input("Type 'yes' or 'no':")
        response = response.lower()
    if response == 'yes':
        train_network = False
        print('Using loaded model')
    else:
        print('training new model')
        best_accuracy = accuracy


    #End of stuff I changed

    if train_network:

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

    r, loop_i, label_orig, label_pert, pert_image = deepfool(example , model)
    
    
    #From deepfool_test.py
    print("Original label = ", label_orig)
    print("Perturbed label = ", label_pert)

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
    plt.imshow(tf(pert_image.cpu()[0]))
    plt.title(label_pert)
    plt.savefig("Image_Original.png")
    plt.show()
    
    #Display original image
    original_image = np.array(example, dtype='float')
    pixels = original_image.reshape((28, 28))
    plt.imshow(pixels)
    plt.savefig("Image_Fooled.png")
    plt.show()
