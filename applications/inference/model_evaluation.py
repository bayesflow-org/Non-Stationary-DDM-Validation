import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

from tqdm import tqdm

import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Palatino"
matplotlib.rcParams['font.family'] = "sans-serif"

from helpers import get_setup
from configurations import model_names

setup = [get_setup(names, "smoothing") for names in model_names]
models = [model[0] for model in setup]
trainers = [trainer[1] for trainer in setup]

NUM_OBS = 768
NUM_SAMPLES = 1000
NUM_RESIMULATIONS = 100

FONT_SIZE_1 = 24
FONT_SIZE_2 = 20
FONT_SIZE_3 = 16

data = pd.read_csv('data/data_color_discrimination.csv')


