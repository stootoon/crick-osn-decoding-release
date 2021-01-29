import os, sys
from glob import glob
from argparse import ArgumentParser
import hashlib

def hasher(folder, classifier):
    to_hash = f"{folder}__{classifier}"
    hashed  = hashlib.md5(to_hash.encode()).hexdigest()
    return hashed, to_hash

if __name__ == "__main__":
    parser = ArgumentParser(description="Submit the jobs to run a specified classifier on an input folder.")
    parser.add_argument("folder",        type=str, help="Folder containing the 'inputs' folder.")
    parser.add_argument("classifier",    type=str, help="Which classifier to use.")
    parser.add_argument("--submit",      help="Whether to submit the jobs.", action="store_true")
    parser.add_argument("--hashonly",    help="Whether to just compute the hashes.", action="store_true")
    parser.add_argument("--missingonly", help="Whether to only run inputs for which the outputs are missing.", action="store_true")
    parser.add_argument("--max",         type = int, default = 100000, help="Maximum number of seeds to run.")
    args = parser.parse_args()
    print(f"Running with {args=}")
    
    hashed, to_hash = hasher(args.folder, args.classifier)
    print(f"{to_hash} -> {hashed}")
    if args.hashonly:
        exit(0)
    prefix = hashed[:4]

    inputs_dir = os.path.join(args.folder, "inputs")
    print(f"Reading inputs from {inputs_dir}")
    input_files = [os.path.split(f)[1] for  f in glob(inputs_dir + "/input*.csv")]
    print(f"Read {len(input_files)} input files.")
    
    outputs_dir = os.path.join(args.folder, args.classifier)
    if not os.path.isdir(outputs_dir):
        print(f"Output folder {outputs_dir} not found.")
        output_files = []
    else:
        print(f"Output folder {outputs_dir} already exists.")
        output_files = [os.path.split(f)[1] for f in glob(outputs_dir + "/output*.csv")]
        print(f"Found {len(output_files)} already in the outputs folder.")

    print(f"Missingonly={args.missingonly}")

    print(f"Running up to {args.max} inputs.")    
    inputs_already_run = [r.replace("output", "input") for r in output_files]
    inputs_to_run = list(set(input_files) - set(inputs_already_run)) if args.missingonly else input_files
    print(f"{len(inputs_to_run)} available inputs to run.")
    inputs_to_run_subset = inputs_to_run[:args.max]
    print(f"{len(inputs_to_run_subset)} to be run after limiting to first {args.max} inputs.")
    if len(inputs_to_run_subset)<10:
        print(inputs_to_run_subset[:10])

    base_dir = os.path.dirname(os.path.realpath(__file__))
    submit_flag = "--submit" if args.submit else ""
    for input_file in inputs_to_run_subset:
        input_id = input_file.replace("input","").replace(".csv","")
        job_name = f"{prefix}{'_'*(4-len(input_id))}{input_id}"
        cmd = f"python $CFDGITPY/cmd2job.py 'python -u {base_dir}/run.py {args.folder}/inputs/{input_file} {args.classifier}' --jobname {job_name} --jobmem 16G {submit_flag}"
        print(cmd)
        os.system(cmd)
    
