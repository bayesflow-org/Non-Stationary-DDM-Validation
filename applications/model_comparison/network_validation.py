import bayesflow as beef
import numpy as np
import tensorflow as tf
import pickle

from tqdm import tqdm
from tensorflow.keras.backend import clear_session

from experiment import ModelComparisonExperiment

import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Palatino"
matplotlib.rcParams['font.family'] = "sans-serif"

FIT_MODEL = False

NUM_MODELS = 4
ENSEMBLE_SIZE = 10
SIMULATION_PER_MODEL = 10000
CHUNCK_SIZE = 200
MODEL_NAMES = [
    'Random walk', 'Mixture random walk',
    'Levy flight', 'Regime switching'
    ]

with open('data/validation_data.pkl', 'rb') as f:
    validation_data = pickle.load(f)

configurator = beef.configuration.DefaultModelComparisonConfigurator(NUM_MODELS)

model_indices = tf.one_hot(np.tile(np.repeat(
    [0, 1, 2, 3], CHUNCK_SIZE), int((SIMULATION_PER_MODEL/CHUNCK_SIZE))), NUM_MODELS
    )

def get_model_probabilities(trainer):
    model_probs = np.zeros((int(SIMULATION_PER_MODEL*NUM_MODELS), NUM_MODELS))
    chunks = np.arange(0, SIMULATION_PER_MODEL+1, CHUNCK_SIZE)
    for i in range(len(chunks)-1):
        sim_1 = {'sim_data': validation_data['model_outputs'][0]['sim_data'][chunks[i]:chunks[i+1]]}
        sim_2 = {'sim_data': validation_data['model_outputs'][1]['sim_data'][chunks[i]:chunks[i+1]]}
        sim_3 = {'sim_data': validation_data['model_outputs'][2]['sim_data'][chunks[i]:chunks[i+1]]}
        sim_4 = {'sim_data': validation_data['model_outputs'][3]['sim_data'][chunks[i]:chunks[i+1]]}

        tmp_validation_data = {
            'model_outputs': [sim_1, sim_2, sim_3, sim_4],
            'model_indices': validation_data['model_indices']
        }

        tmp_validation_data_configured = configurator(tmp_validation_data)
        with tf.device('/cpu:0'):
            model_probs[(chunks[i]*NUM_MODELS):(chunks[i+1]*NUM_MODELS)] = trainer.amortizer.posterior_probs(
                tmp_validation_data_configured
            )
    return model_probs

if __name__ == '__main__':
    if FIT_MODEL:
        model_probs_per_ensemble = np.zeros((ENSEMBLE_SIZE, SIMULATION_PER_MODEL*NUM_MODELS, NUM_MODELS))
        for ensemble in tqdm(range(ENSEMBLE_SIZE)):
            clear_session()
            trainer = ModelComparisonExperiment(
            checkpoint_path=f'checkpoints/ensemble_{ensemble}'
            )
            model_probs_per_ensemble[ensemble] = get_model_probabilities(trainer)

        np.save('data/validation_model_probs_per_ensemble.npy', model_probs_per_ensemble)
    else:
        model_probs_per_ensemble = np.load('data/validation_model_probs_per_ensemble.npy')

    # aggregate over ensembles
    average_model_probs = model_probs_per_ensemble.mean(axis=0)

    cal_curves = beef.diagnostics.plot_calibration_curves(
        true_models=model_indices,
        pred_models=average_model_probs,
        model_names=MODEL_NAMES,
        fig_size=(18, 4),
        title_fontsize=22,
        label_fontsize=20,
        tick_fontsize=18,
        legend_fontsize=18
        )

    cal_curves.savefig("plots/calibration_curves.pdf", dpi=300, bbox_inches="tight")

    confusion_matrix = beef.diagnostics.plot_confusion_matrix(
        model_indices,
        average_model_probs,
        model_names=MODEL_NAMES,
        xtick_rotation=45,
        ytick_rotation=0
        )

    confusion_matrix.savefig("plots/confusion_matrix.pdf", dpi=300, bbox_inches="tight")
