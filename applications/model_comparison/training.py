# Get rid of annoying tf warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
import numpy as np

import bayesflow as beef
import tensorflow as tf

import sys
sys.path.append("../")
from experiment import ModelComparisonExperiment

# gpu setting and checking
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
print(tf.config.list_physical_devices('GPU'))

experiment = ModelComparisonExperiment()

with open('data/training_data.pkl', 'rb') as f:
    training_data = pickle.load(f)

history = experiment.run(
    training_data=training_data
    )
