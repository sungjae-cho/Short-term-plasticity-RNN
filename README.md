# Short-term-plasticity-RNN
Code accompanying the paper:

Circuit mechanisms for the maintenance and manipulation of information in working memory

Nicolas Y Masse, Guangyu R Yang, H. Francis Song, Xiao-Jing Wang, David J Freedman 

https://www.nature.com/articles/s41593-019-0414-3

and

https://www.biorxiv.org/content/early/2018/04/22/305714 (preprint)


Code used to train recurrent neural networks endowed with short-term plasticity on working-memory based tasks.

Requires:  
Python 3  
TensorFlow 1+

Please email masse@uchicago.edu to report any issues, bugs, etc.

## Installation
- You should install `scikit-learn` through miniforge if you use the Apple M1 chip (according to [this scikit-learn document](https://scikit-learn.org/stable/install.html#installing-on-apple-silicon-m1-hardware)).
  The installation command is `conda install scikit-learn`.
  Then, `numpy` and `scipy`, which are dependencies of this repo, are installed together.
- I have installed `matplotlib` through `conda install matplotlib` because I do not want some dependency errors.

## Tutorial

### Descriptions for files

Only key features.

```
analysis.py : Analyze trained networks to obtain measures in the paper.
 ├ def run_multiple : Run analyze_model_from_file for multiple networks.
 ├ def analyze_model_from_file : Analyze a trained network.
 ├ def run_model : Run a trained network.
 └ (rest of functions) : Helpers for running a network and analyzing its results.
model.py
 ├ class Model
 │ ├ def __init__
 │ ├ def initialize_wieghts : Initialize weights in Model.
 │ ├ def run_model : Forward propagation for whole steps.
 │ ├ def rnn_cell : Forward propagation at each step.
 │ └ def optimize : Initialize a weight update operation.
 ├ def main : Training function. Both CPU and GPU modes are available.
 └ (rest of functions) : Helpers.
parameters.py
 ├ par : Where model and environmental parameters are stored.
 └ (rest of functions) : Helpers to initialize and edit par.
run_all_models.py : To train multiple networks to get the same experimental results.
simulate_STP.py : Simulating short-time potentiation (STP) related to STSP.
 ├ def run_simulation : Simulate STP and plot results. See test_STP.ipynb.
 ├ def run_sim_step : Update function for each time step.
 └ def create_stp_constants : Function to define STP parameters.
stimulus.py
 └ class Stimulus : Dataset generator for multiple cognitive tasks.
   └ Tasks : ['DMS','DMRS45','DMRS90','DMRS90ccw','DMRS180','DMC', 'DMS+DMRS',
       'DMS+DMRS_early_cue', 'DMS+DMRS_full_cue', 'DMS+DMC','DMS+DMRS+DMC',
       'location_DMS']
```
- How to train the network: Run `main()` in `model.py`. You can specifying what computer device to use by giving GPU ID as a string and `none` to use CPU.

### Following an experiment in the paper

```
# Run to train 20 networks and save their weights.
# python run_all_models.py <GPU_ID> <project_name> <n_networks> <start_model_index>
python run_all_models.py 1 STSP-EIRNN_DMS_20 10 0
python run_all_models.py 2 STSP-EIRNN_DMS_20 10 10
```
