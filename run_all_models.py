'''
How to run:
python run_all_models.py <GPU_ID> <project_name> <n_networks> <start_model_index>
e.g.
python run_all_models.py 1 STSP-EIRNN_DMS_20 10 0
python run_all_models.py 2 STSP-EIRNN_DMS_20 10 10
'''

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
    project_name = sys.argv[2]
    n_networks = int(sys.argv[3])
    start_model_index = int(sys.argv[4])
    print('Selecting GPU ', gpu_id)
    print('Project name: ', project_name)
    print('N networks to train: ', n_networks)
    print('Model index to start: ', start_model_index)
except:
    gpu_id = None
    print('python run_all_models.py <GPU_ID> <project_name> <n_networks> <start_model_index>')
    exit()


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
    for n in range(start_model_index, start_model_index + n_networks):
        run_name = '{}-run-{}'.format(task, str(n))
        save_fn = run_name + '.pkl'
        update_parameters({'trial_type': task, 'save_fn': save_fn})
        try_model(gpu_id, run_name, project_name)
