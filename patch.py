import torch
from torch.autograd import Variable
import memtorch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from memtorch.utils import LoadMNIST
import numpy as np
import copy
from memtorch.mn.Module import patch_model
from memtorch.map.Input import naive_scale
from memtorch.map.Parameter import naive_map

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
	device = torch.device('cpu' if 'cpu' in memtorch.__version__ else 'cuda')
	batch_size = 256
	train_loader, validation_loader, test_loader = LoadMNIST(batch_size=batch_size, validation=False)


	reference_memristor = memtorch.bh.memristor.VTEAM
	reference_memristor_params = {'time_series_resolution': 1e-10}
	memristor = reference_memristor(**reference_memristor_params)
	memristor.plot_hysteresis_loop()
	memristor.plot_bipolar_switching_behaviour()

	model = Net().to(device)
	model.load_state_dict(torch.load('trained_model.pt'), strict=False)
	patched_model = patch_model(copy.deepcopy(model),
                          memristor_model=reference_memristor,
                          memristor_model_params=reference_memristor_params,
                          module_parameters_to_patch=[torch.nn.Conv2d],
                          mapping_routine=naive_map,
                          transistor=True,
                          programming_routine=None,
                          tile_shape=(128, 128),
                          max_input_voltage=0.3,
                          scaling_routine=naive_scale,
                          ADC_resolution=8,
                          ADC_overflow_rate=0.,
                          quant_method='linear')

	patched_model.tune_()

	print(test(patched_model, test_loader))