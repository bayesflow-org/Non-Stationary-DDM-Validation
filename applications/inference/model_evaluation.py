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
CONDITION_NAMES = ["Accuracy Condition", "Speed Condition"]
BAR_WIDTH = np.arange(-0.6, 0.7, 0.2)
X_AXIS_VALUES = np.arange(4) * 2
LABELS = [
    'Empiric', 'Random walk', 'Mixture random walk',
    'Levy flight', 'Regime switching'
    ]
COLORS = [
    "black", "orange", "maroon", "#133a76", "green"
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

def plot_parameter_trajectory(person_data, local_samples, winning_model, lw=2):
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
            color=COLORS[winning_model+1], alpha=0.9, lw=lw, label="Posterior median"
            )
        ax.fill_between(
            range(NUM_OBS),
            post_median[:, i] - post_mad[:, i],
            post_median[:, i] + post_mad[:, i],
            color=COLORS[winning_model+1], alpha=0.5, label="Posterior MAD"
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
    # fig.suptitle(f'Parameter Trajectory of {MODEL_NAMES[winning_model]}', fontsize=FONT_SIZE_0)
    # legend
    handles = [
        Line2D(xdata=[], ydata=[], color=COLORS[winning_model+1], alpha=0.8, lw=3, label="Posterior median"),
        Patch(facecolor=COLORS[winning_model+1], alpha=0.5, edgecolor=None, label="Posterior MAD"),
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
    
def plot_posterior_resimulation(summaries):
    handles = []
    fig, axarr = plt.subplots(1, 2, figsize=(16, 6))
    for i, ax in enumerate(axarr.flat):
        for t, summary in enumerate(summaries):
            ax.scatter(
                X_AXIS_VALUES + BAR_WIDTH[t],
                summary.loc[summary.speed_condition == i, 'median'],
                s=75, color=COLORS[t], label=LABELS[t]
            )

            ax.errorbar(
                X_AXIS_VALUES + BAR_WIDTH[t],
                summary.loc[summary.speed_condition == i, 'median'],
                yerr=summary.loc[summary.speed_condition == i, 'mad'],
                # fmt='none', capsize=5, elinewidth=2,
                # color=COLORS[t]
                fmt='o', color=COLORS[t], markersize=8, elinewidth=2, capsize=0
                )

            handles.append(
                Line2D(
                    xdata=[], ydata=[], marker='o', markersize=10, lw=3,
                    color=COLORS[t], label=LABELS[t]
                )
            )

        ax.set_title(CONDITION_NAMES[i], fontsize=FONT_SIZE_1)

        x_labels = ['1', '2', '3', '4']
        x_positions = [0, 2, 4, 6]
        ax.set_xticks(x_positions, x_labels)

        ax.set_ylim([
            summaries[0]["median"].min() - 0.1,
            summaries[0]["median"].max() + 0.4])

        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_3)
        if i == 0:
            ax.set_ylabel("Response Time", labelpad=10, fontsize=FONT_SIZE_2)

        ax.set_xlabel("Difficulty", labelpad=10, fontsize=FONT_SIZE_2)

    # legend
    fig.subplots_adjust(hspace=0.5)
    fig.legend(
        handles,
        LABELS,
        fontsize=FONT_SIZE_2, bbox_to_anchor=(0.5, -0.05),
        loc="center", ncol=5
        )
    sns.despine()
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    # compute overall empiric summaries
    grouped_data = data.groupby(['speed_condition', 'difficulty'])
    overall_summary = grouped_data.agg({
                'rt': ['median', lambda x: np.median(np.abs(x - np.median(x)))]
            }).reset_index(drop=False)
    overall_summary.columns = ['speed_condition', 'difficulty', 'median', 'mad']

    summary_per_model = []
    resim_data_per_model = []
    summary_per_model.append(overall_summary)
    for i, model in enumerate(models):
        resim_data = np.zeros((NUM_SUBJECTS, NUM_RESIMULATIONS, NUM_OBS, 3))
        for sub in range(NUM_SUBJECTS):
            # compute indiviudal summaries
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
            f = plot_parameter_trajectory(person_data, local_samples, winning_model)
            f.savefig(f"plots/parameter_trajectory_sub_{sub+1}.pdf", dpi=300, bbox_inches="tight")
            plt.close()
            # posterior re-simulation for all models
            idx = np.random.choice(np.arange(NUM_SAMPLES), NUM_RESIMULATIONS, replace=False)
            pred_data = np.abs(
                model.likelihood(samples_per_model[i]['local_samples'][sub, :, idx, :])['sim_data']
            )
            pred_rt = pred_data[:, :, None]
            condition = np.tile(person_data['speed_condition'], (NUM_RESIMULATIONS, 1))[:, :, None]
            difficulty = np.tile(person_data['difficulty'], (NUM_RESIMULATIONS, 1))[:, :, None]
            resim_data[sub] = np.c_[pred_rt, condition, difficulty]
        
        resim_data_per_model.append(resim_data)
        # overall re-simulation
        reshaped_data = resim_data.reshape(-1, 3)
        df = pd.DataFrame(reshaped_data, columns=['rt', 'speed_condition', 'difficulty'])
        grouped_data = df.groupby(['speed_condition', 'difficulty'])
        summary = grouped_data.agg({
            'rt': ['median', lambda x: np.median(np.abs(x - np.median(x)))]
        })
        summary = summary.reset_index(drop=False)
        summary.columns = ['speed_condition', 'difficulty', 'median', 'mad']
        summary_per_model.append(summary)
        f = plot_posterior_resimulation(summary_per_model)
        f.savefig("plots/post_resimulation_overall.pdf", dpi=300, bbox_inches="tight")
        plt.close()
    # individual re-simulation
    for sub in range(NUM_SUBJECTS):
        # compute indiviudal summaries
        person_data = data.loc[data.id == sub+1]
        grouped = person_data.groupby(['speed_condition', 'difficulty'])
        person_summary = grouped.agg({
            'rt': ['median', lambda x: np.median(np.abs(x - np.median(x)))]
        }).reset_index(drop=False)
        person_summary.columns = ['speed_condition', 'difficulty', 'median', 'mad']
        summaries = []
        summaries.append(person_summary)
        for i in range(len(models)):
            temp_data = resim_data_per_model[i][sub]
            reshaped_data = temp_data.reshape(-1, 3)
            df = pd.DataFrame(reshaped_data, columns=['rt', 'speed_condition', 'difficulty'])
            grouped_data = df.groupby(['speed_condition', 'difficulty'])
            summary = grouped_data.agg({
                'rt': ['median', lambda x: np.median(np.abs(x - np.median(x)))]
            }).reset_index(drop=False)
            summary.columns = ['speed_condition', 'difficulty', 'median', 'mad']
            summaries.append(summary)
        f = plot_posterior_resimulation(summaries)
        f.savefig(f"plots/post_resimulation_{sub+1}.pdf", dpi=300, bbox_inches="tight")
        plt.close()
