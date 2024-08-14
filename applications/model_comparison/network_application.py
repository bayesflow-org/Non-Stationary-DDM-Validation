import bayesflow as beef
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from tqdm import tqdm
from tensorflow.keras.backend import clear_session

from experiment import ModelComparisonExperiment

import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Palatino"
matplotlib.rcParams['font.family'] = "sans-serif"

FIT_MODEL = False

NUM_OBS = 768
NUM_MODELS = 4
ENSEMBLE_SIZE = 10
MODEL_NAMES = [
    'Random\nwalk', 'Mixture\nrandom\nwalk',
    'Levy\nflight', 'Regime\nswitching'
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
        model_probs = np.load('data/empiric_model_probs_per_ensemble.npy')

    # compute proportion winning model
    binary_model_prob = np.zeros(model_probs.shape)
    for i in range(10):
        tmp_mat = model_probs[i]
        binary_model_prob[i][np.arange(tmp_mat.shape[0]), np.argmax(tmp_mat, axis=1)] = 1
    binary_model_prob_per_ensemble = binary_model_prob.mean(axis=1)
    mean_binary_model_prob_per_ensemble = binary_model_prob_per_ensemble.mean(axis=0)
    std_binary_model_prob_per_ensemble = binary_model_prob_per_ensemble.std(axis=0)
    # compute mean log10 bayes factors
    pmps = np.stack(model_probs)
    bayes_factors = np.log10(pmps[:, :, np.newaxis, :] / pmps[:, :, :, np.newaxis])
    mean_bf = np.mean(bayes_factors, axis=(0, 1))

    # plot PMP
    fig, ax = plt.subplots(1, 2, figsize=(9, 3), gridspec_kw={'width_ratios': [0.55, 1]})
    parts = ax[0].violinplot(
        # mean_model_probs
        model_probs.reshape((10*14, 4))
    )
    for pc in parts['bodies']:
        pc.set_facecolor('maroon')
        pc.set_alpha(0.5)

    parts['cbars'].set_color('maroon')
    # parts['cmeans'].set_color('maroon')
    parts['cmaxes'].set_color('maroon')
    parts['cmins'].set_color('maroon')

    ax[0].set_xticks(np.arange(1,5), MODEL_NAMES)

    ax[0].tick_params(axis='both', which='major', labelsize=10)
    ax[0].set_ylabel("Posterior model probability", labelpad=10, fontsize=12)
    ax[0].set_xlabel("", labelpad=10, fontsize=12)
    ax[0].set_ylim(0.0, 1)
    ax[0].grid(alpha=0.3)
    sns.despine()

    ax[1] = sns.heatmap(
        mean_bf,
        cmap=LinearSegmentedColormap.from_list("", ["white", "#8f2727"]),
        annot=True,
        square=True,
        xticklabels=MODEL_NAMES,
        yticklabels=MODEL_NAMES,
        cbar_kws={"shrink": .85},
        annot_kws={"fontsize":9}
    )
    ax[1].axhline(y=0, color='k',linewidth=1.5)
    ax[1].axhline(y=mean_bf.shape[1], color='k',linewidth=1.5)
    ax[1].axvline(x=0, color='k',linewidth=1.5)
    ax[1].axvline(x=mean_bf.shape[0], color='k',linewidth=1.5)

    ax[1].tick_params(axis='both', which='major', labelsize=10)
    ax[1].tick_params(axis='y', which='major', labelsize=10, rotation=0)

    cbar = ax[1].collections[0].colorbar
    cbar.ax.tick_params(labelsize=9)

    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    fig.savefig("plots/model_probabilties.pdf", dpi=300, bbox_inches="tight")
    # determine winning model per person
    binary_model_prob_per_person = binary_model_prob.mean(axis=0)
    winning_model_per_person = np.argmax(binary_model_prob_per_person, axis=1)
    winning_model_per_person[-1] = 2
    file_paths = ['data/winning_model_per_person.pkl', '../inference/data/winning_model_per_person.pkl']
    for file_path in file_paths:
        with open(file_path, 'wb') as file:
            pickle.dump(winning_model_per_person, file)
