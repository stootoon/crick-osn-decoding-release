import os, sys
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import logging
import time

import inputs
from classifiers import classifiers

default_score_function = lambda search: search.score
def regression_prediction_to_score(search):
    return lambda X, y: np.mean(np.sign(search.predict(X) + np.random.randn(*y.shape)*1e-8).astype(int) == np.sign(y).astype(int))

def mock_predictors(X, mock="null"):
    if mock == "null":
        pass
    elif mock == "full_rand":
        print("Mocking data as IID standard normals.")
        X = np.random.randn(*X.shape)
    elif mock == "shuf_cols":
        print("Mocking data by shuffling the columns.")
        X = np.array([np.random.permutation(Xi) for Xi in X.T]).T
    elif mock == "rand_cols":
        print("Mocking data by generating columns with same mean and sd.")
        X = np.array([np.random.randn(*Xi.shape) * np.std(Xi) + np.mean(Xi) for Xi in X.T]).T
    else:
        raise ValueError(f"Don't know what to do for {mock=}")
    return X
    
def run_single(config, search, mock = "null", score_function = default_score_function):
    X, y = inputs.generate_input_for_config(config)

    print(f"Running with {config}.")    

    X = mock_predictors(X, mock)    

    train_scores, test_scores = [], []
    for itrn, itst in StratifiedShuffleSplit().split(X,y):
        X_trn, y_trn = X[itrn], y[itrn]
        X_tst, y_tst = X[itst], y[itst]
        search.fit(X_trn, y_trn)            
        train_scores.append(score_function(search)(X_trn, y_trn))
        test_scores.append(score_function(search)(X_tst, y_tst))
    
    return np.mean(train_scores), np.mean(test_scores)

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

def get_pipeline(classifier=None, raw=False, return_pipe = False, verbose = False, **kwargs):
    clf        = classifiers[classifier].classifier()
    params     = classifiers[classifier].parameters
    
    steps = [] if raw else [('scaler', StandardScaler())]
    steps += [('clf', clf)]

    if verbose:
        print(f"Pipeline {steps=}")
    pipe = Pipeline(steps = steps)

    param_grid = {('clf__'+fld):vals for fld,vals in params.items()}
    if verbose:
        print(f"{param_grid=}")

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
    is_regression  = classifiers[args.classifier].is_regression
    score_function = default_score_function if not is_regression else regression_prediction_to_score

    records = []
    done    = 0
    start_time = time.time()
    logging.getLogger("inputs").setLevel(logging.DEBUG)
    for index, conf in confs.iterrows():
        print(f"*"*120)
        train_score, test_score = run_single(conf, search, mock = args.mock, score_function=score_function)
        print(f"{train_score=}, {test_score=}")
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

    
