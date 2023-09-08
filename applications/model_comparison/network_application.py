import bayesflow as beef
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

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
MODEL_NAMES = [
    'Random walk', 'Mixture random walk',
    'Levy flight', 'Regime switching'
    ]
FONT_SIZE_1 = 24
FONT_SIZE_2 = 20
FONT_SIZE_3 = 16

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

if __name__ == '__main__':
    if FIT_MODEL:
        model_probs_per_ensemble = np.zeros((ENSEMBLE_SIZE, NUM_SUBJECTS, NUM_MODELS))
        for ensemble in tqdm(range(ENSEMBLE_SIZE)):
            clear_session()
            trainer = ModelComparisonExperiment(
                checkpoint_path=f'checkpoints/ensemble_{ensemble}'
            )
            model_probs_per_ensemble[ensemble] = trainer.amortizer.posterior_probs(EMPIRIC_DATA)

        np.save('data/empiric_model_probs_per_ensemble.npy', model_probs_per_ensemble)
    else:
        model_probs_per_ensemble = np.load('data/empiric_model_probs_per_ensemble.npy')

    # compute proportion winning model
    binary_model_prob = np.zeros(model_probs_per_ensemble.shape)
    for i in range(ENSEMBLE_SIZE):
        tmp_mat = model_probs_per_ensemble[i]
        binary_model_prob[i][np.arange(tmp_mat.shape[0]), np.argmax(tmp_mat, axis=1)] = 1
    binary_model_prob_per_ensemble = binary_model_prob.mean(axis=1)
    mean_binary_model_prob_per_ensemble = binary_model_prob_per_ensemble.mean(axis=0)
    std_binary_model_prob_per_ensemble = binary_model_prob_per_ensemble.std(axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(
        MODEL_NAMES,
        mean_binary_model_prob_per_ensemble,
        color='maroon', alpha=1.0
    )
    ax.errorbar(
        MODEL_NAMES,
        mean_binary_model_prob_per_ensemble,
        yerr=std_binary_model_prob_per_ensemble,
        fmt='none', capsize=5, elinewidth=1,
        color='maroon', alpha=0.8
        )

    ax.set_xticks(MODEL_NAMES, MODEL_NAMES, rotation=45, ha="right")
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_3)
    ax.set_ylabel("Average maximal\nmodel probabilty", rotation=0, labelpad=90, fontsize=FONT_SIZE_2)
    ax.set_xlabel("Model", fontsize=FONT_SIZE_2)
    ax.set_ylim(-0.05, 1)
    sns.despine()

    fig.savefig("plots/model_probabilties.pdf", dpi=300, bbox_inches="tight")

    binary_model_prob_per_person = binary_model_prob.mean(axis=0)
    winning_model_per_person = np.argmax(binary_model_prob_per_person, axis=1)
    winning_model_per_person[-1] = 2
    winning_model_per_person = [MODEL_NAMES[i] for i in winning_model_per_person]

    with open('data/winning_model_per_person.pkl', 'wb') as file:
        pickle.dump(winning_model_per_person, file)
