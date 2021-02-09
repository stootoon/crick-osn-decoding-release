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
## Installation
To install the code and data,
1. Download the code repository and unpack at your desired **installation root**.
2. Download the data from [this](https://www.dropbox.com/s/pncq56d4evnx7v4/crick-osn-model-release-data.tar.gz?dl=0) link (~250 MB).
3. Unpack the data file to yield a `data` folder.
4. Move this folder into the **installation root**. It should now sit at the same level as the `README.md` file.
You should now be able to run the Jupyter notebook `make_figures.ipynb` to produce the figures in the paper.
## Data
The data folder contains
1. `data.p`: A Numpy pickle file containing the calcium responses of the 145 glomeruli that were decoded;
   - This file is used internally by the other scripts to assemble data for classification.
   - The data can be accessed directly as a dictionary at Python prompt using:
	 ```
	 import numpy as np;
	 data = np.load("data.p", allow_pickle = True).item()
	 ```
2. `sweeps`: A folder containing the results of the parameter sweeps used in the paper.


## Code Usage
### 1. Generate json files defining sweeps using `create_sweeps.py`. 
- For example: `python create_sweeps.py filt --pairs AB,CD,EF --window_size 0.025,0.1,2 --whiskers yes --template max_glom_sweep.json --tstep 0.2 --response_threshold 1 --min_resp_trials 0.75`
- For example: `python create_sweeps.py full --pairs AB,CD,EF --window_size 2 --whiskers yes`
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
  
