# Overview
This repository contains code to produce **Extended Data panels 8k,l** of *Ackels et al. (under review)*, and additional related panels that were not included in the paper. The panels show the performance of a variety of classifiers when decoding whether the two odours presented were fluctuating in a correlated or anti-correlated way.

The code was written and tested using Python 3.8, Jupyter Notebook 6.0.3 on CentOS Linux 7.6.1810.
# Setup
To install the code and data,
1. Download the code repository and unpack at your desired **installation root**.
2. Download the data from [this](https://www.dropbox.com/s/pncq56d4evnx7v4/crick-osn-model-release-data.tar.gz?dl=0) link (~250 MB).
3. Unpack the data file to yield a `data` folder.
4. Move this folder into the **installation root**. It should now sit at the same level as the `README.md` file.

To create the figures in the paper, run the Jupyter notebook `make_figures.ipynb`.
# Data
The `data` folder contains:
## `data.p`: File containing the calcium responses of the 145 glomeruli used.
- This file is used internally by the other scripts to assemble data for classification.
- The data can be accessed directly as a dictionary at Python prompt using:
  ```python
  import numpy as np;
  data = np.load("data.p", allow_pickle = True).item()
  ```
- The dictionary contains the following fields:
  - `dt`: A scalar, containing the sampling time in seconds;
  - `pairs`: A list of the odour pairs used;
  - `stims`: A dictionary keyed by `2` and `20` (the stimulus frequencies) and contains the names of the stimuli used at that frequency. E.g.
	```python
	>>> data["stims"][2]
	['2Hzanti-corr01', '2Hzanti-corr02', '2Hzcorr01', '2Hzcorr02']
	```
  - `experiments`: A list of the names of the experiments whose data was pooled across;		 
  - `ind2expr`: A 145-element list of tuples indicating which ROI in which experiment each ROI is. For example, to find the origin of the 45th (base 0) ROI,
	```python
	>>> data["ind2expr"][45]
	('2020_11_08_ASBA8.7e', 14)
	```
	indicating that it is ROI 14 (base 0) in experiment 2020_11_08_ASBA8.7e;
  - `X`: A dictionary containing calcium imaging data itself. The first level of the dictionary is keyed by frequency, and the level below by (stimulus, odour_pair) tuples. For example, to determine the shape of the data for the 2Hz anti-correlated stimulus pattern using odour pair AB
	```python
	>>> data["X"][2]['2Hzanti-corr02', 'AB'].shape
	(12, 145, 370)
	```
	- The names of the stimulus condition are those in `data["stims"]` described above.
    - The odour pairs available are those provided in `data["pairs"]` described above.
	- The first dimension are 12 repetitions. The first 6 are with whiskers intact, the last 6 with whiskers clipped. Only the first 6 trials are use for the analyses in the paper.
	- The second dimension are the 145 glomeruli.
	- The last dimension are the 370 time points.
## `sweeps`: A folder containing the results of the parameter sweeps used in the paper.
  - This folder contains JSON files describing each sweep, and corresponding folders containing the results of the sweep.
  - The name of each JSON file and corresponding folder describes the parameters of the run.
  - The names are of the form `[prefix]_[#seeds]x_[odour_pair]_[window_size_in_ms]ms_W[yes|no|both]`.
	- `W[yes|no|both]` indicates which whisker trials were used:
		- `Wyes`: Only trials with intact whiskers were used;
		- `Wno`: Only trials without intact whiskers were used;
		- `Wboth`: All trials were used.
  - The name is suffixed with additional terms if non-default parameter values were used.
	- `dt`: The spacing between decoding windows, in seconds;
	- `rt`: The response threshold in baseline SDs, when filtering for responsivity;
	- `mrt`: The minimum number/fraction of responsive trials, when filtering for responsivity;
  - For example, for the run named `filt_100x_EF_25ms_Wyes_dt0.2_rt1.000_mrt0.75`:
	- `filt`: The base name of the sweeps;
	- `100x`: Accuracy was computed for 100 random seeds;
	- `EF`: The odour pair used was `EF`;
	- `25ms`: A window size close to 25 ms was used;
	- `Wyes`: Only whiskered trials were used;
	- `dt0.2`:The windows were spaced 200 ms apart;
	- `rt1.000`: The response threshold was 1 baseline SD (adjusted for window size);
	- `mrt0.75`: The minimum fraction of responsive trials was 0.75.
  - For the sweeps prefixed `all` the decoding performance of random subsets of glomeruli were measured. The sizes of the subsets used was varied in steps of 10 from 1 to 145. Glomeruli were selected at random i.e. no filtering for responsivity was performed.
  - In the sweeps prefixed `filt` all *responsive* glomeruli were used in each window.
  - Each sweep folder contains:
	1. An `inputs` folder, which contains
		- CSV files `inputXYZ.csv`. These files contain input configurations to be run and are passed to `run.py`.
		- Each row inside and `inputXYZ.csv` file specifies a particular input configuration.
		- The header row describes the meaning of each column, such as the seed being used, the frequency of the input, etc.
		- In addition to the `inputXYZ.csv` files, the `inputs` also contains a pickle file `config_filtering.p` which contains the details of the responsivity filtering procedure used when creating the sweep, that determined which configurations were to be run.
	2. One classifier folder for each classifier used on these inputs.
		- For example the folder `data/sweeps/all_100x_AB_2000ms_Wyes/lasso_lars_no_intercept` contains the results of running the `lasso_lars_no_intercept` classier on the inputs in `data/sweeps/all_100x_AB_2000ms_Wyes/inputs`.
		- Each classifier folder contains a list of `outputXYZ.csv` files, one for each `inputXYZ.csv` file.
		- Each row of an output file corresponds to the same row in the inputs file.
        - The final two columns indicate the training and test accuracy when running the given classifier on the given input configuration.
		- The classifier folders also contain SLURM output files showing a detailed log of the accuracy computation process.
# Code Usage
The full process of creating and running a sweep like those used in the paper are described here. To run the decoding analysis on the inputs provided for the sweeps in `data/sweeps` jump straight to step 3.
## 1. Generate JSON files defining sweeps using `create_sweeps.py`. 
- For example: `python create_sweeps.py simple --pairs AB,CD,EF --window_size 2 --whiskers yes`
- This will create JSON files prefixed with base name `simple` to run sweeps using odours AB, CD and EF, a window size of 2, and using only trials with intact whiskers.
- The remaining sweep parameters will be read from `template_sweep.json`.
- The JSON files for the sweep will be written to `data/sweeps`.
## 2. Generate the input configurations for each of the jobs carrying out the sweep using `inputs.py`
- Once the JSON files specifying a sweep are created, the next step is to create the corresponding `inputXYZ.csv` files.
- The `inputs.py` script is used to do this.
- For example: `python inputs.py data/sweeps/simple_100x_AB_2000ms_Wyes.json 10`
- This generates 10 csv files, named 'input000.csv', 'input001.csv' etc.
- The csv files are written to `data/sweeps/simple_100x_AB_2000ms_Wyes/inputs`
- The rows of each csv file contain all the input configurations for a single job.
## 3. Run a configuration using `run.py`
- Finally, the decoding analysis can be run on the input configurations specified in one of the `inputXYZ.csv` files.
- For example: `python run.py data/sweeps/simple_100x_AB_1000ms_Wboth/inputs/input000.csv lasso_lars_no_intercept`
- This will run the `lasso_lars_no_intercept` classifier on the input configurations listed in `input000.csv`
- The outputs will be written to : `data/sweeps/simple_100x_AB_1000ms_Wboth/lasso_lars_no_intercept`
- The rows of the resulting `outputXYZ.csv` file will contain the training and test accuracy when using the specified classifier on the configuration specified on the corresponding row of the corresponding `inputXYZ.csv` file.
# Other Files
- `classifiers.py`: Contains a list of the classifiers used and ranges for any parameters that were to be optimized, such as the SVM `C` parameter.
- `paths.py`: Contains paths to important directories.
- `submit_sweep.py`: A helper file for running all the inputs in a folder as SLURM jobs.
  - For example: `python submit_sweep.py data/sweeps/sweep_100x_AB_1000ms_Wboth lasso_lars --submit`
  - This will submit jobs to run all the `lasso_lars` classifier on inputs in the specified folder.
- `template_sweep.json`: Contains the default sweep parameters which are then modified by `create_sweep.py` to define individual sweeps. Notably, the number of glomeruli used `n_sub` is set to vary in steps of 10 from 1 to 145. This was used to create the `full` sweep provided.
- `max_glom_sweep.json`: Similar to `template_sweep.json` but the number of glomeruli used `n_sub` is fixed at the maximum value of 145. This was used for the `filt` sweeps provided, in which glomeruli were subsequently filtered for responsivity.
  

<!-- LocalWords:  Ackels Jupyter CentOS stims Hzanti corr Hzcorr JSON filt -->
<!-- LocalWords:  responsivity inputXYZ csv config lars outputXYZ SLURM py -->
<!-- LocalWords:  AB EF json SVM glom -->
