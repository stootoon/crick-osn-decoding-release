import os, sys
import logging

logging.basicConfig()
logger = logging.getLogger("consts.py")
logger.setLevel(logging.DEBUG)

base_dir   = os.path.split(os.path.abspath(__file__))[0]
data_dir   = os.path.join(base_dir, "data")
sweeps_dir = os.path.join(data_dir, "sweeps")

logger.info("{base_dir=}")
logger.info("{data_dir=}")
logger.info("{sweeps_dir=}")


