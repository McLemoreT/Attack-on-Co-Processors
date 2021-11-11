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
from memtorch.bh.nonideality.NonIdeality import apply_nonidealities

def patchIdeals(model, args):
    reference_memristor = memtorch.bh.memristor.VTEAM
    reference_memristor_params = {'time_series_resolution': 1e-10}
    memristor = reference_memristor(**reference_memristor_params)
        
    print("Args: ", args)
    
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
    #print(test(patched_model, test_loader))


    patched_model = apply_nonidealities(copy.deepcopy(patched_model),
                                  non_idealities=[memtorch.bh.nonideality.NonIdeality.DeviceFaults],
                                  lrs_proportion=0.25,
                                  hrs_proportion=0.10,
                                  electroform_proportion=0)

    #print(test(patched_model_, test_loader))


    patched_model = apply_nonidealities(copy.deepcopy(patched_model),
                                  non_idealities=[memtorch.bh.nonideality.NonIdeality.Endurance],
                                  x=1e4,
                                  endurance_model=memtorch.bh.nonideality.endurance_retention_models.model_endurance_retention,
                                  endurance_model_kwargs={
                                        "operation_mode": memtorch.bh.nonideality.endurance_retention_models.OperationMode.sudden,
                                        "p_lrs": [1, 0, 0, 0],
                                        "stable_resistance_lrs": 100,
                                        "p_hrs": [1, 0, 0, 0],
                                        "stable_resistance_hrs": 1000,
                                        "cell_size": 10,
                                        "temperature": 350,
                                  })

    #print(test(patched_model_, test_loader))



    patched_model = apply_nonidealities(copy.deepcopy(patched_model),
                                  non_idealities=[memtorch.bh.nonideality.NonIdeality.Retention],
                                  time=1e3,
                                  retention_model=memtorch.bh.nonideality.endurance_retention_models.model_conductance_drift,
                                  retention_model_kwargs={
                                        "initial_time": 1,
                                        "drift_coefficient": 0.1,
                                  })

    #print(test(patched_model, test_loader))



    patched_model = apply_nonidealities(copy.deepcopy(patched_model),
                                  non_idealities=[memtorch.bh.nonideality.NonIdeality.FiniteConductanceStates],
                                  conductance_states=5)

    #print(test(patched_model, test_loader))



    patched_model = apply_nonidealities(copy.deepcopy(patched_model),
                                  non_idealities=[memtorch.bh.nonideality.NonIdeality.NonLinear],
                                  simulate=True)

    #print(test(patched_model, test_loader))



#    patched_model = apply_nonidealities(copy.deepcopy(patched_model),
#                                  non_idealities=[memtorch.bh.nonideality.NonIdeality.NonLinear],
#                                  sweep_duration=2,
#                                  sweep_voltage_signal_amplitude=1,
#                                  sweep_voltage_signal_frequency=0.5)

    #print(test(patched_model, test_loader))
    sigma = 10 #FIXME

    reference_memristor = memtorch.bh.memristor.VTEAM
    reference_memristor_params = {'time_series_resolution': 1e-10,
                              'r_off': memtorch.bh.StochasticParameter(loc=1000, scale=200, min=2),
                              'r_on': memtorch.bh.StochasticParameter(loc=5000, scale=sigma, min=1)}

    memristor = reference_memristor(**reference_memristor_params)
    memristor.plot_hysteresis_loop()
    memristor.plot_bipolar_switching_behaviour()
    
    return patched_model