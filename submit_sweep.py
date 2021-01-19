import os, sys
from argparse import ArgumentParser
import hashlib

parser = ArgumentParser(description="Submit the jobs to run a specified classifier on an input folder.")
parser.add_argument("folder",        type=str, help="Folder containing the 'inputs' folder.")
parser.add_argument("classifier",    type=str, help="Which classifier to use.")
parser.add_argument("--submit",      help="Whether to submit the jobs.", action="store_true")
args = parser.parse_args()

to_hash = f"{args.folder}__{args.classifier}"
hashed  = hashlib.md5(to_hash.encode()).hexdigest()
print(f"{to_hash} -> {hashed}")
prefix = hashed[:4]
cmd = "ls FOLDER/inputs | grep -Po Q[0-9]+Q | xargs -I{} sh -c Qpython $CFDGITPY/cmd2job.py 'python -u run.py FOLDER/inputs/input{}.csv CLASSIFIER' --jobname PREFIX_{} --jobmem 16G SUBMITQ"
cmd = cmd.replace("FOLDER", args.folder)
cmd = cmd.replace("CLASSIFIER", args.classifier)
cmd = cmd.replace("PREFIX", prefix)
cmd = cmd.replace("SUBMIT", "--submit" if args.submit else "")
cmd = cmd.replace('Q','"')
print(cmd)
os.system(cmd)

