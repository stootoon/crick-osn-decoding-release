# Decoding OSN Calcium Signals
This repository contains code to produce some of the OSN decoding figures in Ackels et al. (under review).
## Usage
### 1. Generate json files defining sweeps using `create_sweeps.py`. 
- For example: `python create_sweeps.py sweep --pairs AB,CD --whiskers yes,no,both`
- This will create json files prefixed with 'sweep' for each combination of pairs and whiskers specified.
- The remaining sweep parameters will be read from `template_sweep.json`.
- The json files for the sweep will be written to `data/sweeps`.
### 2. Generate the input configurations for each of the jobs carrying out the sweep using `inputs.py`
- For example: `python inputs.py data/sweeps/sweep_100x_AB_1000ms_Wboth.json 100`
- This generates up to 100 csv files, named 'input000.csv', 'input001.csv' etc.
- The csv files are written to `data/sweeps/sweep_100x_AB_1000ms_Wboth/inputs`
- The rows of each csv file contain all the input configurations for a single job.
### 3. Run a configuration using `run.py`
- For example: `python run.py data/sweeps/sweep_100x_AB_1000ms_Wboth/inputs/input000.csv lasso_lars`
- This will run the `lasso_lars` classifier on the input configurations listed in `input000.csv`
- The outputs will be written to : `data/sweeps/sweep_100x_AB_1000ms_Wboth/lasso_lars`
## Misc Files
- `template_sweep.json`: Contains the default sweep parameters which are then modified to define individual sweeps.
- `consts.py`: Contains various constants, like paths to data directories, etc.
- `classifiers.list`: A list of the classifiers available
- `submit_sweep.py`: A helper file for running all the inputs in a folder as SLURM jobs.
  - For example: `python submit_sweep.py data/sweeps/sweep_100x_AB_1000ms_Wboth lasso_lars --submit`
  - This will submit jobs to run all the `lasso_lars` classifer on inputs in the specified folder.
  
