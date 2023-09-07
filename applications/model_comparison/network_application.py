import bayesflow as beef
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from tensorflow.keras.backend import clear_session

from experiment import ModelComparisonExperiment

import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Palatino"
matplotlib.rcParams['font.family'] = "sans-serif"

FIT_MODEL = True

NUM_OBS = 768
NUM_MODELS = 4
ENSEMBLE_SIZE = 10

configurator = beef.configuration.DefaultModelComparisonConfigurator(NUM_MODELS)

# prepare empirical data
data = pd.read_csv('data/data_color_discrimination.csv')
data['rt'] = np.where(data['correct'] == 0, -data['rt'], data['rt'])
NUM_SUBJECTS = len(np.unique(data['id']))
emp_data = np.zeros((NUM_SUBJECTS, NUM_OBS, 1), dtype=np.float32)
for i in range(NUM_SUBJECTS):
    tmp = data[data['id'] == i+1]
    emp_data[i] = tmp['rt'].to_numpy()[:, np.newaxis]
EMPIRIC_DATA = {'summary_conditions': emp_data}

def get_model_probabilities(data, trainer):
    return trainer.amortizer.posterior_probs(data)

if __name__ == '__main__':
    if FIT_MODEL:
        model_probs_per_ensemble = np.zeros((ENSEMBLE_SIZE, NUM_SUBJECTS, NUM_MODELS))
        for ensemble in tqdm(range(ENSEMBLE_SIZE)):
            clear_session()
            trainer = ModelComparisonExperiment(
                checkpoint_path=f'checkpoints/ensemble_{ensemble}'
            )
            model_probs_per_ensemble[ensemble] = get_model_probabilities(EMPIRIC_DATA, trainer)

        np.save('data/empiric_model_probs_per_ensemble.npy', model_probs_per_ensemble)
    else:
        model_probs_per_ensemble = np.load('data/empiric_model_probs_per_ensemble.npy')
