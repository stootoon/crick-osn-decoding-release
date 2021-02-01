import os, sys
from collections import namedtuple
import numpy as np
from numpy import vstack
from builtins import sum as bsum
import pandas as pd
import pathlib
import pickle
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

dataset_name = 'CvsAC_OSN_imaging_awake_Sina'
dataroot     = os.path.join(os.getenv('DATA'), "tobias", dataset_name, 'unpacked')

from consts import data_dir

Config = namedtuple('Config', 'seed n_sub shuf freq pairs whiskers window_size start_time')

def load_experiments():
    from scipy.io import loadmat as loadmat_    
    loadmat  = lambda *args: loadmat_(os.path.join(dataroot, *args))
    
    meta = loadmat("meta.mat")["meta"]
    meta = {n:meta[n].flatten()[0][0][0] for n in meta.dtype.names}
    logger.info(f"{meta=}")
    
    fs = meta["acqRate"]
    dt = 1/meta["acqRate"]
    logger.info(f"{fs=:1.3f}, {dt=:1.3f}")
    logger.info(f"Odour on at {dt*meta['pulseONInt']:1.3f}")
    
    pairs = sorted(["AB", "CD", "EF"])
    logger.info(f"{pairs=}")
    
    experiments = sorted([d.name for d in os.scandir(dataroot) if d.is_dir()])
    logger.info(f"{experiments=}")
    
    stims = sorted([d.name for d in os.scandir(os.path.join(dataroot, experiments[0], "AB")) if d.is_dir()])
    logger.info(f"{stims=}")
    
    valid_stims = {freq:[stim for stim in stims if f"{freq}Hz" in stim and "corr" in stim] for freq in [2,20]}
    logger.info(f"{valid_stims=}")
    
    data = {(experiment, pair, stim):loadmat(os.path.join(experiment, pair, stim, "Xgrt.mat"))["Xgrt"] for experiment in experiments for pair in ["AB", "CD", "EF"] for stim in stims}
    
    X = {freq:{(stim, pair):vstack([data[expr,pair,stim] for expr in experiments]).transpose([1,0,2]) for pair in pairs for stim in vstims} for freq, vstims in valid_stims.items()}
    
    ind2expr = bsum([[(expr, i) for i in range(data[expr, pairs[0], valid_stims[2][0]].shape[0])] for expr in experiments], [])
    
    stim_pairs = list(X[2].keys())
    n_reps, n_glom, n_time = X[2][stim_pairs[0]].shape
    logger.info(f"{n_reps=}, {n_glom=}, {n_time=}")

    data_file = os.path.join(data_dir, "data.p")
    with open(data_file, "wb") as f:
        np.save(f, {"dt":dt, "X":X, "pairs":pairs, "experiments":experiments, "ind2expr":ind2expr, "stims":valid_stims}, allow_pickle=True)
    logger.info(f"Wrote {data_file}.")

def validate_pairs(pairs):
    if not all([p in ["AB", "CD", "EF"] for p in pairs]):
        raise ValueError("Found invalid pairs in {pairs=}")
    return pairs

whisker_slice = {"yes":slice(0,6), "no":slice(6,12), "both":slice(0,12)}

def filter_by_responsivity(Xrgt, t_all, ind_t_resp, t_baseline = [0, 3], response_threshold = 0, min_resp_trials = 0, return_full = False):
    logger.debug("filter_by_responsivity")
    Xrgt_bl   = np.copy(Xrgt[:,:,(t_all >= t_baseline[0]) & (t_all < t_baseline[-1])])

    # Reshape the array so that I can compute stats over reps and time
    Xgtr_bl   = Xrgt_bl.transpose([1,2,0])
    Xg_tr_bl  = Xgtr_bl.reshape(Xgtr_bl.shape[0],-1)

    # Compute the baseline stats per glomerulus
    g_mean_bl = np.mean(Xg_tr_bl, axis=1)
    g_std_bl  = np.std(Xg_tr_bl,  axis=1)
    
    # Compute the responses by averaging over the desired trials
    Xrg_resp  = np.mean(Xrgt[:,:,ind_t_resp],axis=-1)

    # Normalize the responses by what they would be assuming baseline stats.
    # The mean is the baseline mean, but sd is scaled by the number of time points
    # we average over to compute the responses.
    n_resp    = len(ind_t_resp)    
    Zrg_resp  = (Xrg_resp - g_mean_bl[np.newaxis,:])/(g_std_bl[np.newaxis,:]/np.sqrt(n_resp))
    logger.debug(f"Z-score range: {np.min(Zrg_resp):1.3f} - {np.max(Zrg_resp):1.3f}.")
    # Count the number of responsive reps per glomerulus
    g_n_resp  = np.sum(np.abs(Zrg_resp) > response_threshold, axis=0)
    logger.debug(f"Found {np.sum(g_n_resp>0)} glomeruli with at least one responsive trial.")

    # Figure out how many response reps (aka trials) we need
    n_trials  = Xrgt_bl.shape[0]
    logger.debug(f"Received {min_resp_trials=}")
    if type(min_resp_trials) is float:
        min_resp_trials = int(min_resp_trials * n_trials)
    logger.debug(f"Interpreted as {min_resp_trials=} trials.")        
    assert 0 <= min_resp_trials <= n_trials, f"{min_resp_trials=} was not between 0 and {n_trials}"

    # The responsive cells are those which responded above threshold
    # on at least min_resp_trials trials.
    flag_resp = g_n_resp>=min_resp_trials
    logger.debug(f"Flagging the {np.sum(flag_resp)} glomeruli with at least {min_resp_trials} responsive trials.")
    
    if return_full:
        return flag_resp, {"baseline_means":g_mean_bl, "baseline_sds":g_std_bl, "trial_responses":Xrg_resp, "trial_responses_norm":Zrg_resp, "n_responsive_trials":g_n_resp, "min_resp_trials":min_resp_trials}
    else:
        return flag_resp

def generate_input_for_config(config, data_file = os.path.join(data_dir, "data.p"), baseline = [0,3], response_threshold = 0, min_resp_trials = 0, return_full = False):
    logger.debug("generate_input_for_config")    
    if not os.path.exists(data_file):
        logger.debug(f"Could not find {data_file=}. Creating.")
        load_experiments()
        if not os.path.exists(data_file):
            raise FileExistsError("Attempted to create {data_file=} but file still not found.")
    
    data  = np.load(data_file, allow_pickle = True).item()
    X, dt, experiments, stims, all_pairs = [data.get(f) for f in ["X", "dt", "experiments", "stims", "pairs"]]

    logger.debug(f"Generating input for {config}")
    np.random.seed(config.seed)

    pairs = validate_pairs([config.pairs[i:i+2] for i in range(0, len(config.pairs),2)])
    logger.debug(f"Using {pairs=}.")

    Xrgt = vstack([X[config.freq][stim,pair][whisker_slice[config.whiskers]] for stim in stims[config.freq] for pair in pairs])
    n_reps, n_glom, n_time = Xrgt.shape
    n_whisker = (lambda x: x.stop - x.start)(whisker_slice[config.whiskers])

    t = np.arange(0, n_time) * dt

    n_bins_per_window = int(np.ceil(config.window_size / dt))
    logger.debug(f"Using {n_bins_per_window} bins to approximate a desired window size of {config.window_size:1.3f} with {dt=:1.3f}.")
    first_bin= np.where(t>=config.start_time)[0][0]
    last_bin = first_bin + n_bins_per_window - 1
    ind_t    = np.arange(first_bin, last_bin + 1).astype(int)
    logger.debug(f"Picked time indices {ind_t[0]} ({t[ind_t[0]]:1.3f} sec.) to {ind_t[-1]} ({t[ind_t[-1]]:1.3f} sec.) to approximately span a {config.window_size=:1.3f} second window starting at {config.start_time=:1.3f} seconds.")

    # Filter glomeruli by responsivity
    logger.debug(f"Filtering glomeruli by responsivity using {response_threshold=} and {min_resp_trials=}.")
    flag_resp     = filter_by_responsivity(Xrgt, t, ind_t, response_threshold = response_threshold, min_resp_trials = min_resp_trials)

    ind_resp_glom = np.where(flag_resp)[0]
    n_resp_glom   = len(ind_resp_glom)
    logger.debug(f"Found {n_resp_glom} responsive glomeruli: {list(ind_resp_glom)}")

    if n_resp_glom > 0:
        if config.n_sub > n_resp_glom:
            logger.warning(f"Number of responsive glomeruli {n_resp_glom} < desired number of glomeruli {config.n_sub}.")
        
        ind_glom = sorted(np.random.permutation(ind_resp_glom)[:config.n_sub])
        logger.debug(f"Picked {len(ind_glom)}/{n_resp_glom=} responsive glomeruli: {ind_glom=}")
       
        X_sub = np.copy(Xrgt[:,ind_glom, :])
    
        X = np.mean(X_sub[:, :, ind_t],axis=-1)    
        logger.debug(f"Shape of predictors: {X.shape=}")
    else:
        logger.warning("No responsive glomeruli found.")
        ind_glom = []
        X = np.array([])
        X_sub = np.array([])
        
    y = []
    [logger.debug((y.append([-1 if "anti-corr" in stim else 1] * n_whisker), f"({stim=}, {pair=}) -> {y[-1]}")[1]) for stim in stims[config.freq] for pair in pairs]    
    y = np.array(y).flatten()
    logger.debug(f"Labels: {list(y)}")

    return (X, y) if not return_full else (X, y, t, X_sub, ind_glom, ind_t)

def compute_gloms_available_for_config(config, **kwargs):
    X = generate_input_for_config(config, **kwargs)[0]
    return X.shape[1] if len(X) else 0        

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("sweep_file", help="JSON file containing the sweep parameters.", type = str)
    parser.add_argument("n_jobs",     help="Number of jobs to create.",                  type = int)
    parser.add_argument("--startat",  help="ID of the first input csv.", default = 0,    type = int)    
    args = parser.parse_args()

    if not os.path.exists(args.sweep_file):
        raise FileNotFoundError(f"Could not find {sweep_file=}.")
    
    sweep_name = args.sweep_file.split(".")[0]
    print(f"Creating {sweep_name=}")

    import json
    
    print(f"Loading sweep file {args.sweep_file}")
    with open(args.sweep_file, "r") as f:
        sweep = json.load(f)        
    print(f"{sweep=}")
    
    if "n_seeds" in sweep:
        first_seed = sweep["first_seed"] if "first_seed" in sweep else 0
        sweep["seed"] = list(range(first_seed, first_seed+sweep["n_seeds"]))
        print(f"Sweeping for seeds {sweep['seed'][0]} - {sweep['seed'][-1]}.")
        
    missing_fields = [fld for fld in Config._fields if fld not in sweep]
    if len(missing_fields):
        raise ValueError(f"The following required fields were missing from the sweep file: {missing_fields=}")

    from itertools import product
    
    fields = Config._fields
    confs  = list(product(*[sweep[fld] for fld in fields]))
    n_confs= len(confs)
    
    print(f"{n_confs} configurations in sweep before filtering.")
    # Each config specifies an n_sub, the desired number of glomeruli to use.
    # But after filtering, we get n_sub_avail.
    # We shouldn't run a config again if its parameters match a previously run one
    # and both have the same n_sub_avail.

    all_confs    = []          
    confs_map    = {}
    confs_to_run = []
    gloms_avail  = {}
    messages     = {}
    for conf_item in confs:
        conf = Config(**{fld:conf_item[i] for i,fld in enumerate(fields)})
        all_confs.append(conf)
        if conf_item not in confs_map:
            key = (conf.n_sub, conf.pairs, conf.freq, conf.whiskers, conf.window_size, conf.start_time)
            if key not in gloms_avail:
                gloms_avail[key] = compute_gloms_available_for_config(conf,
                                             response_threshold = sweep["response_threshold"],
                                             min_resp_trials    = sweep["min_resp_trials"],
                                             return_full = False)
            n_glom_avail    = gloms_avail[key]
            confs_map[conf] = conf._replace(n_sub = n_glom_avail)
        if confs_map[conf] not in confs_to_run:
            confs_to_run.append(confs_map[conf])
            messages[conf] = f"{conf} maps to {confs_map[conf]}, which will be added to the run list."
        else:
            messages[conf] = f"{conf} maps to {confs_map[conf]}, which is already in the run list."
        print(messages[conf])
        
    n_confs_to_run = len(confs_to_run)
    print(f"{n_confs_to_run}/{n_confs} to be run.")
    
    folder, json_file = os.path.split(os.path.abspath(args.sweep_file))
    folder = os.path.join(folder, json_file[:-5], "inputs")
    print(f"Writing to {folder=}")
    os.makedirs(folder, exist_ok = True)
    
    n_confs_per_job = n_confs_to_run // args.n_jobs
    print(f"{n_confs_per_job=}")
    
    for i, istart in enumerate(range(0,n_confs_to_run, n_confs_per_job)):
        df = pd.DataFrame(confs_to_run[istart:istart+n_confs_per_job], columns=fields)
        df["response_threshold"] = sweep["response_threshold"]
        df["min_resp_trials"]    = sweep["min_resp_trials"]        
        output_file = os.path.join(folder, f"input{i + args.startat:03d}.csv")
        df.to_csv(output_file, index=False)

    print(f"Wrote {i+1} input files to {folder=}")

    # We run into issues if we try to pickle Namedtuples, so recast
    all_confs = [item._asdict() for item in all_confs]
    Config2tuple = lambda conf: tuple([conf._asdict()[fld] for fld in fields])
    confs_map = {Config2tuple(key):Config2tuple(val) for key,val in confs_map.items()}
    confs_to_run = [item._asdict() for item in confs_to_run]
    messages  = {Config2tuple(key):msg for key,msg in messages.items()}
    conf_file = os.path.join(folder, "config_filtering.p")
    config_filtering = {"all_confs":all_confs,
                        "confs_map":confs_map,
                        "confs_to_run":confs_to_run,
                        "messages":messages,
                        "gloms_avail":gloms_avail,
                        "fields":fields}    
    with open(conf_file, "wb") as f:
        pickle.dump(config_filtering, f)
    print(f"Wrote config filtering data structures to {conf_file}.")

    print("ALLDONE")

    
    
    

