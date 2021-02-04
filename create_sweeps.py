import os, sys
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser(description="Generate json files for various sweeps.")
parser.add_argument("base_name", type=str, help="The base name to use for the sweeps.")
parser.add_argument("--nseeds", type=int, default=100, help="Number of seeds to use.")
parser.add_argument("--firstseed", type=int, default=0, help="First seed value.")
parser.add_argument("--pairs", type=str, default="AB,CD,EF", help="Comma separted list of which pairs to use.")
parser.add_argument("--window_size", type=str, default="2", help="Comma separated list of which window sizes to use.")
parser.add_argument("--whiskers", type=str, default="yes",help="Comma separated list of which whiskers to use.")
parser.add_argument("--template", type=str, default="template_sweep.json",help="The template JSON file to base the results on.")
parser.add_argument("--tfirst", type=float, default=0, help="Start of the first window.")
parser.add_argument("--tlast", type=float, default=10, help="Start of the last window.")
parser.add_argument("--tstepmin", type=float, default=0.5, help="Minimum window step size.")
parser.add_argument("--tstep", type=float, default=-1, help="Set positive to directly specify a stepsize.")
parser.add_argument("--response_threshold", type=float, default=0, help="Response threshold for filtering glomeruli.")
parser.add_argument("--min_resp_trials",    type=float, default=0, help="Minimum responsive trials for filtering glomeruli.")
args = parser.parse_args()
print(args)

import json

from paths import sweeps_dir

with open(args.template, "rb") as f:
    template = json.load(f)

if not os.path.isdir(sweeps_dir):
    print(f"Folder {sweeps_dir} not found, creating.")
    os.makedirs(sweeps_dir, exist_ok = True)

# min_resp_trials is interpretd as a fraction if float or a number of trials if int.
# We have to accept it as a float above, but if it's close to the int value, assume it's int
if np.abs(args.min_resp_trials - int(args.min_resp_trials)) <1e-6:
    print(f"Casting {args.min_resp_trials=} to ",end="")
    args.min_resp_trials = int(args.min_resp_trials)
    print(f"{args.min_resp_trials=}")
    
resp_filtering_suffix = ""
if args.response_threshold:
    resp_filtering_suffix += f"_rt{args.response_threshold:1.3f}"
if args.min_resp_trials:
    resp_filtering_suffix += f"_mrt{args.min_resp_trials}"
    
for pairs in args.pairs.split(","):
    for window_size in args.window_size.split(","):
        wnd   = float(window_size)
        tstep = args.tstep if (args.tstep > 0) else max(args.tstepmin, wnd/2)
        for whiskers in args.whiskers.split(","):            
            sweep = dict(template)
            sweep["response_threshold"] = [args.response_threshold]
            sweep["min_resp_trials"]    = [args.min_resp_trials]
            sweep["n_seeds"]     = args.nseeds
            sweep["first_seed"]  = args.firstseed
            sweep["pairs"]       = [pairs]
            sweep["window_size"] = [wnd]
            sweep["whiskers"]    = [whiskers]
            sweep["start_time"] = list(np.arange(args.tfirst, args.tlast, tstep))
            suffix = "_".join([str(args.nseeds) + "x", pairs, f"{int(wnd*1000)}ms", f"W{whiskers}"])
            if args.tstep > 0:
                suffix += f"_dt{args.tstep}"

            name   = args.base_name + "_" + suffix + resp_filtering_suffix

            file_name = os.path.join(sweeps_dir, f"{name}.json")
            with open(f"{file_name}", "w") as f:
                json.dump(sweep, f)
            print(f"Wrote {file_name}.")
            


