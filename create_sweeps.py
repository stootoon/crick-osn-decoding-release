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
args = parser.parse_args()
print(args)

import json

from consts import sweeps_dir

with open(args.template, "rb") as f:
    template = json.load(f)

if not os.path.isdir(sweeps_dir):
    print(f"Folder {sweeps_dir} not found, creating.")
    os.makedirs(sweeps_dir, exist_ok = True)
    
for pairs in args.pairs.split(","):
    for window_size in args.window_size.split(","):
        wnd   = float(window_size)
        tstep = max(args.tstepmin, wnd/2)
        for whiskers in args.whiskers.split(","):            
            sweep = dict(template)
            sweep["n_seeds"]     = args.nseeds
            sweep["first_seed"]  = args.firstseed
            sweep["pairs"]       = [pairs]
            sweep["window_size"] = [wnd]
            sweep["whiskers"]    = [whiskers]
            sweep["start_time"] = list(np.arange(args.tfirst, args.tlast, tstep))
            suffix = "_".join([str(args.nseeds) + "x", pairs, f"{int(wnd*1000)}ms", f"W{whiskers}"])
            name   = args.base_name + "_" + suffix

            file_name = os.path.join(sweeps_dir, f"{name}.json")
            with open(f"{file_name}", "w") as f:
                json.dump(sweep, f)
            print(f"Wrote {file_name}.")
            


