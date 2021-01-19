import os, sys
import logging

logging.basicConfig()
logger = logging.getLogger("classify")
logger.setLevel(logging.INFO)

base_dir = os.path.split(os.path.abspath(__file__))[0]

data_dir   = os.path.join(base_dir, "data")
sweeps_dir = os.path.join(data_dir, "sweeps")


