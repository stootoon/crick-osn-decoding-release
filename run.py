import os, sys
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import time

import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

import inputs
from classifiers import classifiers, score_function

def mock_predictors(X, mock="null"):
    if mock == "null":
        pass
    elif mock == "full_rand":
        logger.info("Mocking data as IID standard normals.")
        X = np.random.randn(*X.shape)
    elif mock == "shuf_cols":
        logger.info("Mocking data by shuffling the columns.")
        X = np.array([np.random.permutation(Xi) for Xi in X.T]).T
    elif mock == "rand_cols":
        logger.info("Mocking data by generating columns with same mean and sd.")
        X = np.array([np.random.randn(*Xi.shape) * np.std(Xi) + np.mean(Xi) for Xi in X.T]).T
    else:
        raise ValueError(f"Don't know what to do for {mock=}")
    return X
    
def run_single(config, search, score_function, mock = "null", response_threshold = 0, min_resp_trials = 0):
    logger.info(f"Running with {config}.")

    X, y = inputs.generate_input_for_config(config, response_threshold=response_threshold, min_resp_trials=min_resp_trials)
    if len(X) == 0:
        logger.warning("No predictors found.")
        return np.nan, np.nan
    
    np.random.seed(config.seed)
    
    X = mock_predictors(X, mock)    

    train_scores, test_scores = [], []
    for i, (itrn, itst) in enumerate(StratifiedShuffleSplit(random_state = config.seed).split(X,y)):
        X_trn, y_trn = X[itrn], np.copy(y[itrn])
        X_tst, y_tst = X[itst], np.copy(y[itst])

        if config.shuf:
            y_trn = np.random.permutation(y_trn)
            y_tst = np.random.permutation(y_tst)

        search.fit(X_trn, y_trn)            
        train_scores.append(score_function(search)(X_trn, y_trn))
        test_scores.append(score_function(search)(X_tst, y_tst))
        logger.info(f"Split {i:>2d}: TRAINING ({sum(y_trn>0):2d}/{len(y_trn):<2d} trials are +1) {train_scores[-1]:1.3f}\tTEST ({sum(y_tst>0):>2d}/{len(y_tst):<2d} trials are +1) {test_scores[-1]:1.3f}")        

    mean_train = np.mean(train_scores)
    mean_test  = np.mean(test_scores)
    std_train  = np.std(train_scores)
    std_test   = np.std(test_scores)
    
    logger.info(f"Train {mean_train:1.3f} +/- {std_train:1.3f}")
    logger.info(f" Test {mean_test:1.3f} +/- {std_test:1.3f}")    

    return mean_train, mean_test

def get_output_folder_name(args, head = None):
    raw           = args["raw"]           if type(args) is dict else args.raw
    mock          = args["mock"]          if type(args) is dict else args.mock
    classifier    = args["classifier"]    if type(args) is dict else args.classifier

    if head is None:
        input_file    = args["input_file"]    if type(args) is dict else args.input_file    
        head, _ = os.path.split(input_file)
        if not head.endswith("inputs"):
            raise ValueError(f"Input file base directory {head=} path must be 'inputs'")
        head, _ = os.path.split(head) # Split again to get the parent of inputs

    folder = ""
    if raw:
        folder += "_raw"
    if mock != "null":
        folder += f"_{mock}"
    folder += "_"+classifier
    if folder[0] == "_":
        folder = folder[1:]
    folder = os.path.join(head, folder)
    return folder

def get_pipeline(classifier, raw=False, return_pipe = False, verbose = False, seed=0, **kwargs):
    clf        = classifiers[classifier].classifier(**({"random_state":seed} if "svc" in classifier else {}))
    params     = classifiers[classifier].parameters
    
    steps = [] if raw else [('scaler', StandardScaler())]
    steps += [('clf', clf)]

    if verbose:
        logger.info(f"Pipeline {steps=}")
    pipe = Pipeline(steps = steps)

    param_grid = {('clf__'+fld):vals for fld,vals in params.items()}
    if verbose:
        logger.info(f"{param_grid=}")

    search = GridSearchCV(pipe, param_grid, n_jobs=-1)
    return (search, pipe) if return_pipe else search
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_file",      help="File to read the input configurations from.", type=str)
    parser.add_argument("classifier",      help="The name of the classifier to use.", type=str)
    parser.add_argument("--output_folder", help="Specific output folder to use.", type=str, default=None)
    parser.add_argument("--raw",           help="Whether to use the raw inputs, instead of standardizing.", action="store_true")
    parser.add_argument("--mock",          help="Whether to mock the predictors. [null|shuf_cols|rand_cols|full_rand]. Default is 'null'.", type=str, default="null")
    args = parser.parse_args()
    print(args)

    valid_mocks = ["null","shuf_cols", "rand_cols", "full_rand"]
    if args.mock not in valid_mocks:
        raise ValueError(f"{args.mock=} is not one of {valid_mocks}.")

    if args.classifier not in classifiers:
        raise ValueError(f"No classifier '{args.classifier}' available.")
            
    output_folder = args.output_folder or get_output_folder_name(args)
    print(f"Writing outputs to {output_folder=}")
    
    if not os.path.exists(args.input_file):
        raise FileNotFoundError("Could not find input file {args.input_file}.")
    confs = pd.read_csv(args.input_file)
    n_confs = len(confs)
    print(f"Read {n_confs} configurations from {args.input_file}.")
    
    search         = get_pipeline(classifier = args.classifier, raw = args.raw, verbose = True)

    records = []
    done    = 0
    start_time = time.time()
    logging.getLogger("inputs").setLevel(logging.DEBUG)
    for index, conf in confs.iterrows():
        print(f"*"*120)
        train_score, test_score = run_single(conf, search, score_function[args.classifier], mock = args.mock)
        print(f"{train_score=:1.3f}")
        print(f" {test_score=:1.3f}")
        new_record = conf.to_dict()
        new_record.update({"train_score":train_score, "test_score":test_score})
        records.append(new_record)
        done += 1
        elapsed_time = time.time() - start_time
        unit_time =  elapsed_time / done
        remaining = unit_time * (n_confs - done)
        print(f"{done=}/{n_confs=}: {elapsed_time=:1.3f} {remaining=:1.3f}")

    df = pd.DataFrame(records)
    print(df.head())
    input_dir, input_file = os.path.split(args.input_file)
    output_file      = input_file.replace("input", "output")
    output_full_path = os.path.join(output_folder, output_file)
    os.makedirs(output_folder, exist_ok=True)
    df.to_csv(output_full_path, index=False)
    print(f"Wrote results to {output_full_path}.")

    print("ALLDONE")

    
