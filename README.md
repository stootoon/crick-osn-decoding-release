# Introduction
This repository contains code to produce **Extended Data panels 8k,l** of *Ackels et al. (under review)*, and additional related panels that were not included in the paper. The panels show the performance of a variety of classifiers when decoding whether the two odours presented were fluctuating in a correlated or anti-correlated way.

The code was written and tested using Python 3.8, Jupyter Notebook 6.0.3 on CentOS Linux 7.6.1810.
<!-- ## Classfication Procedure -->
<!-- The aim of the classification was to determine whether the two odours presented to the animal were fluctuating in a correlated or anticorrelated manner using the calcium responses of 145 glomeruli sampled at 30 Hz from 3 seconds before odour onset to 9 seconds after odour onset.  -->
<!-- ### Classifier Inputs -->
<!-- Odours were presented in pairs. Three pairs of odours ('AB','CD', and 'EF') were used with concentrations fluctuating according to whether -->
<!-- - Fluctuations were at 2 Hz or 20 Hz; -->
<!-- - The two odours fluctuated in-sync ('correlated') or out-of-sync ('anti-correlated'); -->
<!-- - The initial phase shift of the first odour was 0 or 180 degrees; -->
<!-- For each setting of the parameters above 6 trials were recorded. This yielded, for each odour pair and frequency, 24 trials for classification: (6 trials) x (2 correlation patterns) x (2 phase shifts). -->

<!-- Independent classifiers were learned for each odour pair, fluctuation frequency, time point of interest and response window size (which we collectively term a 'configuration'). Response window sizes used were 1 bin (~33 ms), 2 bins (~66 ms), 4 bins (~132 ms), and 62 bins (~2 seconds). The response of each of the 145 ROIs to each stimulus was averaged over the relevant bins at each time point. This yielded, for each each configuration, a matrix of 24 samples x 145 predictors, where each sample contained the responses of the 145 ROIs to one of the 24 stimuls trials for that configuration, and a corresponding 24-dimensional vector of labels whose elements were +1 if the corresponding trial had correlated fluctuations, and -1 if anti-correlated. Finally, before classification, the predictors matrix was standardized so that columns had mean zero and unit variance. -->

<!-- See the function `get_input_for_config` in `inputs.py` for the relevant code. -->
<!-- ### Choice of classifiers -->
<!-- Because we had fewer samples than predictors, the data was linearly separable and we used regularized classifiers to promote the learning of robust classification boundaries. We used off-the-shelf classifiers provided by scikit-learn. -->

<!-- We began by using support vector classifiers with linear kernels. We started with the standard l2 penalty on the weights. While this gave good classification results, we were also aiming for the intepretability. Although the l2 penalty promotes small weights it usually does not set any to zero, implicating all ROIs in every classification. To get more intepretable results, we switched to using the l1 penalty on the weights. This gave similar classification performance but the resulting sparse weight vectors allowed us to more easily find and verify the ROIs contributing to a given classification performance.  -->

<!-- We initially also learned intercepts for these classifiers, but found that this led to overfitting as evidenced by sub-chance shuffled performance, so we subsequently held intercepts at zero. This resulted in chance-level performance for the shuffled trials, as we expected. -->

<!-- Support vector classifiers have a parameter C which must be tuned to get good performance. We performed this tuning by performing a grid search over a fixed set of powers of 10. But we also able to get equally good classification performance and interpretability by using the Lasso while also avoiding the manual tuning of the C parameter by using the lasso in the `LassoLarsCV' incarnation provided by scikit-learn. Because the lasso is technically a regression procedure, to use it as a classifier we added a very small amount of random noise to its predicted outputs for each trial and took the sign of the result as the classification prediction. The additive noise was to force the selection of a random sign whenever the lasso has learned the all-zeros weight vector, for which the prediction for each trial would otherwise be exactly zero. Thus because it does not require parameter tuning, provides good classification performance and interpretabile weights we ultimately settled on the Lasso when computing decoding accuracies. -->

<!-- #### Nonlinear  -->
## Setup
To install the code and data,
1. Download the code repository and unpack at your desired **installation root**.
2. Download the data from [this](https://www.dropbox.com/s/pncq56d4evnx7v4/crick-osn-model-release-data.tar.gz?dl=0) link (~250 MB).
3. Unpack the data file to yield a `data` folder.
4. Move this folder into the **installation root**. It should now sit at the same level as the `README.md` file.

To create the figures in the paper, run the Jupyter notebook `make_figures.ipynb`.
## Data
The `data` folder contains:
### `data.p`: A Numpy pickle file containing the calcium responses of the 145 glomeruli.
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
### `sweeps`: A folder containing the results of the parameter sweeps used in the paper.
  - This folder contains JSON files describing each sweep, and corresponding folders containing the results of the sweep.
  - The name of each JSON file and corresponding folder describes the parameters of the run.
  - The names are of the form `[prefix]_[#seeds]x_[odour_pair]_[window_size_in_ms]ms_W[yes|no|both]`.
	- `W[yes|no|both]` indicates which whisker trials were used:
		- `Wyes`: Only trials with intact whiskers wer used;
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
		- In addition to the `inputXYZ.csv` files, the `inputs` also contains a pickle file `config_filtering.p` which continains the details of the responsivity filtering procedure used when creating the sweep, that determined which configurations were to be run.
	2. One classifier folder for each classifier used on these inputs.
		- For example the folder `data/sweeps/all_100x_AB_2000ms_Wyes/lasso_lars_no_intercept` contains the results of running the `lasso_lars_no_intercept` classier on the inputs in `data/sweeps/all_100x_AB_2000ms_Wyes/inputs`.
		- Each classifier folder contains a list of `outputXYZ.csv` files, one for each `inputXYZ.csv` file.
		- Each row of an output file corresponds to the same row in the inputs file.
        - The final two columns indicate the training and test accuracy when running the given classifier on the given input configuration.
		- The classifier olders also contain SLURM output files showing a detailed log of the accuracy computation process.
## Code Usage
The full process of creating and running a sweep like those used in the paper are described here. To run the decoding analysis on the inputs provided for the sweeps in `data/sweeps` jump straight to step 3.
### 1. Generate JSON files defining sweeps using `create_sweeps.py`. 
- For example: `python create_sweeps.py simple --pairs AB,CD,EF --window_size 2 --whiskers yes`
- This will create JSON files prefixed with base name `simple` to run sweeps using odours AB, CD and EF, a window size of 2, and using only trials with intact whiskers.
- The remaining sweep parameters will be read from `template_sweep.json`.
- The JSON files for the sweep will be written to `data/sweeps`.
### 2. Generate the input configurations for each of the jobs carrying out the sweep using `inputs.py`
- Once the JSON files specifying a sweep are created, the next step is to create the correspoding `inputXYZ.csv` files.
- The `inputs.py` script is used to do this.
- For example: `python inputs.py data/sweeps/sweep_100x_AB_1000ms_Wboth.json 100`
- This generates up to 100 csv files, named 'input000.csv', 'input001.csv' etc.
- The csv files are written to `data/sweeps/sweep_100x_AB_1000ms_Wboth/inputs`
- The rows of each csv file contain all the input configurations for a single job.
### 3. Run a configuration using `run.py`
- Finally, the decoding analysis can be run on the input configurations specified in one of the `inputXYZ.csv` files.
- For example: `python run.py data/sweeps/sweep_100x_AB_1000ms_Wboth/inputs/input000.csv lasso_lars`
- This will run the `lasso_lars` classifier on the input configurations listed in `input000.csv`
- The outputs will be written to : `data/sweeps/sweep_100x_AB_1000ms_Wboth/lasso_lars`
- The rows of the resulting `outputXYZ.csv` file will contain the training and test accuracy when using the specified classifier on the configuration specified on the corresponding row of the corresponding `inputXYZ.csv` file.
## Misc Files
- `template_sweep.json`: Contains the default sweep parameters which are then modified to define individual sweeps.
- `consts.py`: Contains various constants, like paths to data directories, etc.
- `classifiers.list`: A list of the classifiers available
- `submit_sweep.py`: A helper file for running all the inputs in a folder as SLURM jobs.
  - For example: `python submit_sweep.py data/sweeps/sweep_100x_AB_1000ms_Wboth lasso_lars --submit`
  - This will submit jobs to run all the `lasso_lars` classifer on inputs in the specified folder.
  
