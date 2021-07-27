import numpy as np
from parameters import *
import model
import sys


def try_model(gpu_id, run_name, project_name):
    try:
        # Run model
        model.main(gpu_id, project_name, run_name)
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt')

try:
    gpu_id = sys.argv[1]
    print('Selecting GPU ', gpu_id)
except:
    gpu_id = None


update_parameters({ 'simulation_reps'       : 0,
                    'batch_size'            : 1024,
                    'learning_rate'         : 0.02,
                    'noise_rnn_sd'          : 0.5,
                    'noise_in_sd'           : 0.1,
                    'num_iterations'        : 2000,
                    'spike_regularization'  : 'L2',
                    'synaptic_config'       : 'full',
                    'test_cost_multiplier'  : 2.,
                    'balance_EI'            : True,
                    'savedir'               : './savedir/'})

task_list = ['DMS']

for task in task_list:
    for n in range(20):
        save_fn = task + str(n) + '.pkl'
        update_parameters({'trial_type': task, 'save_fn': save_fn})
        try_model(gpu_id)
