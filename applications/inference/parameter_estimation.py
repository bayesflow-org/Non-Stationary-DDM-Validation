import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

from tqdm import tqdm
from tensorflow.keras.backend import clear_session

from helpers import get_setup
from configurations import model_names

NUM_SAMPLES = 2000
NUM_OBS = 768

data = pd.read_csv('data/data_color_discrimination.csv')
data['rt'] = np.where(data['correct'] == 0, -data['rt'], data['rt'])

NUM_SUBJECTS = len(np.unique(data['id']))
emp_data = np.zeros((NUM_SUBJECTS, NUM_OBS, 1), dtype=np.float32)
for i in range(NUM_SUBJECTS):
    tmp = data[data['id'] == i+1]
    emp_data[i] = tmp['rt'].to_numpy()[:, np.newaxis]

def model_inference(model, trainer):
    num_local_params = model.local_prior_means.shape[0]
    num_hyper_params = model.hyper_prior_means.shape[0]
    local_samples_z = np.zeros((NUM_SUBJECTS, NUM_OBS, NUM_SAMPLES, num_local_params))
    hyper_samples_z = np.zeros((NUM_SUBJECTS, NUM_SAMPLES, num_hyper_params))
    with tf.device('/cpu:0'):
        for i in range(NUM_SUBJECTS):
            clear_session()
            subject_data = {'summary_conditions': emp_data[i:i+1]}
            samples = trainer.amortizer.sample(subject_data, NUM_SAMPLES)
            local_samples_z[i] = samples['local_samples']
            hyper_samples_z[i] = samples['global_samples']

    local_samples = local_samples_z * model.local_prior_stds + model.local_prior_means
    hyper_samples = hyper_samples_z * model.hyper_prior_stds + model.hyper_prior_means

    return {'local_samples': local_samples, 'hyper_samples': hyper_samples}

if __name__ == '__main__':
    samples_per_model = []
    for model_name in tqdm(model_names):
        model, trainer = get_setup(model_name, "smoothing")
        samples = model_inference(model, trainer)
        samples_per_model.append(samples)
    with open('data/posteriors/samples_per_model.pkl', 'wb') as file:
        pickle.dump(samples_per_model, file)
