import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from tqdm import tqdm
from scipy.stats import median_abs_deviation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Palatino"
matplotlib.rcParams['font.family'] = "sans-serif"

from helpers import get_setup
from configurations import model_names

setup = [get_setup(names, "smoothing") for names in model_names]
models = [model[0] for model in setup]
# trainers = [trainer[1] for trainer in setup]

NUM_OBS = 768
NUM_SAMPLES = 1000
NUM_RESIMULATIONS = 100

LOCAL_PARAM_LABELS = ['Drift rate', 'Threshold', 'Non-decision time']
LOCAL_PARAM_NAMES  = [r'v', r'a', r'\tau']
MODEL_NAMES = [
    'Random walk DDM', 'Mixture random walk DDM',
    'Levy flight DDM', 'Regime switching DDM'
    ]

BAR_WIDTH = np.arange(-0.2, 0.3, 0.1)
X_AXIS_VALUES = np.arange(4) * 2
LABELS = [
    'Empiric', 'Random walk', 'Mixture random walk',
    'Levy flight', 'Regime switching'
    ]
FONT_SIZE_0 = 26
FONT_SIZE_1 = 24
FONT_SIZE_2 = 20
FONT_SIZE_3 = 16

# read empiric data, winning model per person, and posterior samples
data = pd.read_csv('data/data_color_discrimination.csv')
NUM_SUBJECTS = len(np.unique(data['id']))
with open('data/posteriors/samples_per_model.pkl', 'rb') as file:
    samples_per_model = pickle.load(file)
with open('data/winning_model_per_person.pkl', 'rb') as file:
    winning_model_per_person = pickle.load(file)

def plot_parameter_trajectory(person_data, local_samples, model_name, lw=2):
    # get conditions
    condition = person_data['speed_condition'].to_numpy()
    idx_speed = []
    if condition[0] == 1:
        idx_speed.append([0])
        idx_speed.append(np.where(condition[:-1] != condition[1:])[0])
        idx_speed = np.concatenate(idx_speed)
    else:
        idx_speed.append(np.where(condition[:-1] != condition[1:])[0])
        idx_speed.append([NUM_OBS])
        idx_speed = np.concatenate(idx_speed)
    # calculate posterior median and mad
    post_median = np.median(local_samples, axis=1)
    post_mad = median_abs_deviation(local_samples, axis=1)
    # plot
    fig, axarr = plt.subplots(3, 1, figsize=(18, 14))
    for i, ax in enumerate(axarr.flat):
        # parameter trajectory
        ax.plot(
            range(NUM_OBS),
            post_median[:, i], 
            color='maroon', alpha=0.9, lw=lw, label="Posterior median"
            )
        ax.fill_between(
            range(NUM_OBS),
            post_median[:, i] - post_mad[:, i],
            post_median[:, i] + post_mad[:, i],
            color='maroon', alpha=0.5, label="Posterior MAD"
            )

        # yellow shades
        x = 0
        while x < idx_speed.shape[0]:
            ax.axvspan(idx_speed[x] + 1, idx_speed[x + 1] + 1, alpha=0.2, color='#f0c654', label="Speed condition")
            x = x + 2
        # difficulty manipulation
        if i == 0:
            ax.plot(
                range(NUM_OBS),
                (person_data['difficulty'] - 3) * -2,
                color='black', alpha=0.5, lw=lw, label="Difficulty manipulation"
                )
        # aestehtics
        ax.set_title(f'{LOCAL_PARAM_LABELS[i]} (${LOCAL_PARAM_NAMES[i]}$)', fontsize=FONT_SIZE_1)
        ax.grid(alpha=0.3)
        time = np.arange(0, 768+1, 48)
        time[0] = 1
        ax.set_xticks(time)
        ax.margins(x=0.01)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_3)
        ax.set_ylabel("Parameter\nValue", rotation=0, labelpad=70, fontsize=FONT_SIZE_2)
        if i == 2:
            ax.set_xlabel("Trial", labelpad=20, fontsize=FONT_SIZE_2)   

    sns.despine()
    # fig.tight_layout()
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle(f'Parameter Trajectory of {model_name}', fontsize=FONT_SIZE_0)
    # legend
    handles = [
        Line2D(xdata=[], ydata=[], color='maroon', alpha=0.8, lw=3, label="Posterior median"),
        Patch(facecolor='maroon', alpha=0.5, edgecolor=None, label="Posterior MAD"),
        Patch(facecolor='#f0c654', alpha=0.2, edgecolor=None, label="Speed condition"),
        Line2D(xdata=[], ydata=[], color='black', alpha=0.5, lw=3, label="Difficulty condition")
        ]
    fig.legend(
        handles,
        ["Posterior median", "Posterior MAD", "Speed condition", "Difficulty condition"],
        fontsize=FONT_SIZE_2, bbox_to_anchor=(0.5, -0.001),
        loc="center", ncol=4
        )
    
    return fig
    
def posterior_resimulation(summaries):



if __name__ == '__main__':
    for sub in range(NUM_SUBJECTS):
        person_data = data.loc[data.id == sub+1]
        grouped = data.groupby(['speed_condition', 'difficulty'])
        person_summary = grouped.agg({
            'rt': ['median', lambda x: np.median(np.abs(x - np.median(x)))]
        })
        person_summary = person_summary.reset_index(drop=False)
        person_summary.columns = ['speed_condition', 'difficulty', 'median', 'mad']

        # parameter trajectory of winning model
        winning_model = winning_model_per_person[sub]
        local_samples = samples_per_model[winning_model]['local_samples'][sub]
        f = plot_parameter_trajectory(person_data, local_samples, MODEL_NAMES[winning_model])
        f.savefig(f"plots/parameter_trajectory_sub_{sub+1}.pdf", dpi=300, bbox_inches="tight")
        
        # posterior re-simulation for all models
        idx = np.random.choice(np.arange(NUM_SAMPLES), NUM_RESIMULATIONS, replace=False)
        summaries = []
        summaries.append(person_summary)
        for i, model in enumerate(models):
            pred_data = np.abs(
                model.likelihood(samples_per_model[i]['local_samples'][sub, :, idx, :])['sim_data']
                )
            pred_df = pd.DataFrame({
                'speed_condition': np.tile(person_data['speed_condition'], NUM_RESIMULATIONS),
                'difficulty': np.tile(person_data['difficulty'], NUM_RESIMULATIONS),
                'rt': pred_data.flatten(),
                })
            grouped = pred_df.groupby(['difficulty', 'speed_condition'])
            pred_summary = grouped.agg({
                'rt': ['median', lambda x: np.median(np.abs(x - np.median(x)))]
            })
            pred_summary = pred_summary.reset_index(drop=False)
            pred_summary.columns = ['difficulty', 'speed_condition', 'median', 'mad']
            summaries.append(pred_summary)
        



