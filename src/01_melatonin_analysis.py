# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 12:46:11 2025

@author: malonso
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns

from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
from scipy.spatial import distance_matrix

os.chdir(Path().absolute())


# %% INPUT

main_data_labels = ["melatonin_flicker", "melatonin_monocular"]
pilot_data_labels = ["melatonin_pilot"]


# %% FUNCTIONS
def import_and_process(path, file_suffix):

    # read file
    df = pd.read_parquet(path.parent / f"{path.stem}_{file_suffix}.parquet")

    # remove irrelevant conditions
    df = df[df["condition"].isin(["dark", "constant"])]

    # rename constant condition
    df.loc[df["condition"] == "constant", "condition"] = "bright light"

    # add data label
    df["data label"] = path.stem.split("_")[1]

    # determine if pupil dilation
    df["pupil_dilation"] = "melatonin_monocular" in path.stem

    return df


def plot_over_time_per_session(
    data_frame: pd.DataFrame,
    x_var: str,
    y_var: str,
    light_exposure: tuple,
    habitual_bedtime: float,
    x_tick_values: list,
    x_tick_labels: list,
    condition_order: list,
    condition_colors: list,
    condition_var: str = "condition",
    x_label: str = "relative time (hh:mm)",
    y_label: str = "",
    fig_title: str = "",
    add_legend: bool = True,
    ax=None,
):

    # get y-axis limits
    y_lim_max = np.nanmax(data_frame[y_var])
    y_lim_min = np.nanmin(data_frame[y_var])

    # Map conditions to custom colors
    palette = dict(zip(condition_order, condition_colors))

    # create figure handle
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # HABITUAL BEDTIME
    ax.axvline(
        x=habitual_bedtime,
        color=[0.5, 0.5, 0.5],
        linestyle="--",
        linewidth=0.9,
    )

    # LIGHT EXPOSURE
    # calculate light exposure patch width
    light_width = light_exposure[1] - light_exposure[0]

    # Add light exposure rectangle to plot
    rect = Rectangle(
        xy=(light_exposure[0], y_lim_min),
        width=light_width,
        height=y_lim_max - y_lim_min,
        color=(1, 0.972549, 0.861745, 0.5),
    )

    ax.add_patch(rect)

    # CONDITIONS

    if add_legend:
        leg = "full"
    else:
        leg = False

    sns.lineplot(
        x=x_var,
        y=y_var,
        hue=condition_var,
        hue_order=condition_order,
        data=data_frame,
        linewidth=2.5,
        ax=ax,  # Pass the existing ax to seaborn
        legend=leg,
        palette=palette,  # Custom color palette
    )

    # X-TICKS AND X-LABELS
    ax.xaxis.set_ticks(x_tick_values)
    ax.set_xticklabels(x_tick_labels)

    # TITLE & LABELS
    ax.set_title(fig_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # LEGEND
    if add_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            sns.move_legend(ax, "upper left")

    return fig, ax


def plot_auc_with_percent(
    data_frame: pd.DataFrame,
    x_var,
    y_var,
    label_var,
    condition_order,
    condition_colors,
    x_label: str = "",
    y_label: str = "",
    fig_title: str = "",
    ax=None,
):

    # create color palette
    palette = dict(zip(condition_order, condition_colors))

    # create figure handle
    if ax is None:
        fig, ax = plt.subplots()

    else:
        fig = ax.figure

    sns.barplot(
        data=data_frame,
        x=x_var,
        y=y_var,
        hue=x_var,
        order=condition_order,
        palette=palette,
        saturation=0.9,
        ax=ax,
        legend=False,
    )

    label_text = [f"{x:.1f}%" for x in data_frame[label_var]]

    for container, label in zip(ax.containers, label_text):
        if container:
            ax.bar_label(container, labels=[label], fontsize=16)

    ax.set_title(fig_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)

    return fig, ax


def add_adaptive_jitter(
    data_frame,
    x_var,
    y_var,
    jitter: tuple = (-0.05, 0.05),
    spread_factor: float = 0.05,
    min_distance: float = 0.01,  # Minimum distance to enforce
    max_iterations: int = 20,  # To prevent infinite adjustment loops
):
    y_distances = distance_matrix(data_frame[[y_var]], data_frame[[y_var]])

    # Inverse scaling: More jitter for closer points
    scale_factor = 1 / (y_distances.sum(axis=1) + 1e-3)
    scale_factor = (scale_factor - scale_factor.min()) / (
        scale_factor.max() - scale_factor.min()
    )

    # Initial jitter
    base_jitter = np.random.uniform(jitter[0], jitter[1], size=len(data_frame))
    offset_spread = (
        np.linspace(-spread_factor, spread_factor, len(data_frame))
        - spread_factor / 2
    )
    np.random.shuffle(offset_spread)

    # Apply base jitter and spread
    x_jittered = data_frame[x_var] + base_jitter * scale_factor + offset_spread

    # Iteratively adjust points to avoid overlap
    x_jittered = x_jittered.to_numpy()  # Convert to numpy array
    for _ in range(max_iterations):
        x_distances = distance_matrix(x_jittered[:, None], x_jittered[:, None])

        # Find pairs closer than the minimum distance
        np.fill_diagonal(x_distances, np.inf)  # Ignore self-distances
        too_close = np.any(x_distances < min_distance, axis=1)

        # Apply additional jitter to problematic points
        if np.any(too_close):
            extra_jitter = np.random.uniform(
                -spread_factor / 2, spread_factor / 2, size=too_close.sum()
            )
            x_jittered[too_close] += extra_jitter
        else:
            break

    data_frame["x_jittered"] = x_jittered
    return data_frame["x_jittered"]


def plot_connected_raincloud(
    data_frame: pd.DataFrame,
    y_var: str,
    x_order: list,
    x_colors: list,
    ax=None,
    x_var: str = "condition",
    connect_var: str = "",
    x_subgroup_var: str = "",
    x_subgroup_palette: str = "mako",
    x_subgroup_order: list = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    offset: float = -0.4,
    box_width: float = 0.2,
    jitter: tuple = (-0.05, 0.05),
    spread_factor: float = 0.05,
    raw_marker: str = "o",
    raw_marker_size: float = 64,
    base_fontsize: float = 22,
    violin_alpha: float = 1,
    fig_size: tuple = (),
):

    # Set up the plot
    if not ax:
        if not fig_size:
            if x_subgroup_var:
                fig, ax = plt.subplots(figsize=(7, 6))
            else:
                fig, ax = plt.subplots(figsize=(7, 5))
        else:
            fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure

    # Re-order categorical variable
    data_frame[x_var] = pd.Categorical(data_frame[x_var], x_order)
    data_frame = data_frame.sort_values(x_var)

    # add jitter for raw data points and lines
    data_frame["x_index"] = data_frame[x_var].cat.codes
    data_frame["x_jittered"] = add_adaptive_jitter(
        data_frame=data_frame,
        x_var="x_index",
        y_var=y_var,
        jitter=jitter,
        spread_factor=spread_factor,
    )

    # Violin plot

    # Plot the half violin plot
    sns.violinplot(
        data=data_frame,
        x=x_var,
        y=y_var,
        order=x_order,
        inner=None,
        linewidth=1.5,
        cut=0,
        # palette=x_colors,
        width=0.6,
        zorder=1,
        split=True,
        ax=ax,
    )

    # color violins after plotting to avoid mirroring
    for i, violin in enumerate(ax.collections[: len(x_order)]):
        violin.set_edgecolor("none")
        r, g, b = sns.color_palette(x_colors)[i]
        violin.set_facecolor((r, g, b, violin_alpha))

        path = violin.get_paths()[0]
        path.vertices[:, 0] += offset  # Apply offset to x-coordinates

    # Boxplot
    # Plot the boxplot
    sns.boxplot(
        data=data_frame,
        x=x_var,
        y=y_var,
        fliersize=0,
        order=x_order,
        showcaps=False,
        width=box_width,
        boxprops={"facecolor": "none", "linewidth": 2.5},
        whiskerprops={"linewidth": 2.5},
        medianprops={"color": "black", "linewidth": 2.5},
        capprops={"linewidth": 2.5},
        zorder=3,
        ax=ax,
    )

    # Connecting lines
    # Plot the raw data points and connection lines
    if connect_var:
        for _, group in data_frame.groupby(connect_var):

            ax.plot(
                group["x_jittered"],
                group[y_var],
                color="grey",
                alpha=0.5,
                linewidth=1,
                zorder=2,
            )

    # Raw data points
    # determine subgroup colour mapping
    if x_subgroup_var:

        # re-order groups according to provided order
        if x_subgroup_order:
            data_frame["x_subgroup_var_cat"] = pd.Categorical(
                data_frame[x_subgroup_var], x_subgroup_order
            )
            data_frame = data_frame.sort_values(x_subgroup_var)
        else:
            x_subgroup_order = data_frame[x_subgroup_var].unique()

        # determine groups and colour palette
        subgroups = data_frame[x_subgroup_var].unique()
        subgroup_mapping = dict(
            zip(
                subgroups,
                sns.color_palette(x_subgroup_palette, len(subgroups)),
            )
        )
        data_frame["subgroup_color"] = (
            data_frame[x_subgroup_var].str.strip().map(subgroup_mapping)
        )

    else:
        # create condition colors
        color_mapping = dict(zip(x_order, x_colors))
        data_frame["condition_color"] = data_frame[x_var].map(color_mapping)

    ax.scatter(
        data_frame["x_jittered"],
        data_frame[y_var],
        c=(
            data_frame["subgroup_color"]
            if x_subgroup_var
            else data_frame["condition_color"]
        ),
        ec="k",
        s=raw_marker_size,
        zorder=4,
        marker=raw_marker,
    )

    # Adjust legend for subgroup coloring
    if x_subgroup_var:
        # Create legend handles with round markers
        legend_handles = [
            mlines.Line2D(
                [],
                [],
                marker="o",
                color=subgroup_mapping[subgroup],
                linestyle="None",
                markersize=10,
                label=subgroup,
            )
            for subgroup in x_subgroup_order
            if subgroup in subgroup_mapping
        ]

        ax.legend(
            handles=legend_handles,
            title=x_subgroup_var,
            bbox_to_anchor=(0.5, -0.15),
            loc="upper center",
            ncol=len(legend_handles),
            columnspacing=0.1,
            handletextpad=-0.3,
        )

    # %% Format

    # Adjust x-axis limits to prevent cutting off violins
    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min + offset, x_max)

    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(ticks=range(len(x_order)), labels=x_order)
    ax.set_title(title)

    return fig, ax


# %% ANALYSIS

# %% Import data

# mange paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.abspath(os.path.join(current_dir, "..", "data"))
figs_path = os.path.abspath(os.path.join(current_dir, "..", "figures"))
os.makedirs(figs_path, exist_ok=True)

# import data
data_dir = f"{data_path}/VR_paper_melatonin_processed data.parquet"
df = pd.read_parquet(data_dir)

auc_dir = f"{data_path}/VR_paper_melatonin_results.parquet"
auc_df = pd.read_parquet(auc_dir)


# %% Prep for plots

grouped = df.groupby("participant_id")
grouped_auc = auc_df.groupby("participant_id")

# time plot parameters
x_var = "sample_number"
light_exposure = (5.2, 9)
habitual_bedtime = 9
x_tick_values = np.arange(1, 12, 2)
x_tick_labels = [
    "-04:00",
    "-03:00",
    "-02:00",
    "-01:00",
    "00:00",
    "01:00",
]
condition_var = "condition"
x_label = "relative time (hh:mm)"
conditions_order = ["dark", "bright light"]
conditions_color = ["dimgrey", "gold"]


# %% Plot raw data
sns.set_theme(font_scale=1.1, style="white")

for participant, group in grouped:

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # SUBPLOT 1 - RAW MELATONIN v TIME
    _, axs[0] = plot_over_time_per_session(
        data_frame=group,
        x_var=x_var,
        y_var="melatonin_concentration",
        light_exposure=light_exposure,
        habitual_bedtime=habitual_bedtime,
        x_tick_values=x_tick_values,
        x_tick_labels=x_tick_labels,
        condition_order=conditions_order,
        condition_colors=conditions_color,
        condition_var=condition_var,
        x_label=x_label,
        y_label="Melatonin concentration (pg/mL)",
        fig_title="Salivary melatonin",
        ax=axs[0],
    )

    # get auc data
    group_auc = grouped_auc.get_group(participant)

    # SUBPLOT 3 - AUC AND SUPP % RAW
    _, axs[1] = plot_auc_with_percent(
        data_frame=group_auc,
        x_var="condition",
        y_var="auc_light",
        label_var="mel_supp",
        condition_order=conditions_order,
        condition_colors=conditions_color,
        y_label="AUC",
        ax=axs[1],
        fig_title="Melatonin suppression",
    )

    # Add main title
    fig.suptitle(
        f"Participant {participant}",
        # y=1.02,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1.05])
    plt.show()

    # Save
    output_file = f"{figs_path}/mel_{participant}_VR_paper_light"
    # fig.savefig(f"{output_file}.png", dpi=500)
    if participant == 137:
        fig.savefig(f"{output_file}.svg", format="svg")
    plt.close(fig)


# %% Plot AUC and supp %
sns.set_theme(font_scale=1.2, style="white")

auc_df["Pupil state"] = auc_df["pupil_dilation"].map(
    {False: "natural", True: "dilated"}
)

fig = plt.figure(constrained_layout=True, figsize=(8, 5.5), dpi=600)
axes = fig.subplot_mosaic("AAB")

plot_connected_raincloud(
    ax=axes["A"],
    data_frame=auc_df,
    y_var="auc_light",
    x_order=conditions_order,
    x_colors=conditions_color,
    x_var="condition",
    connect_var="participant_id",
    x_subgroup_var="Pupil state",
    x_subgroup_palette="muted",
    title="",
    x_label="condition",
    y_label="AUC",
    jitter=(-0.01, 0.03),
    spread_factor=0.03,
    offset=-0.42,
    box_width=0.2,
    violin_alpha=0.8,
    raw_marker_size=45,
)
axes["A"].set_title(
    "Cumulative melatonin concentration during HMD wear", pad=10
)
axes["A"].get_legend().remove()

# Melatonin suppression

plot_connected_raincloud(
    ax=axes["B"],
    data_frame=auc_df[auc_df["condition"] == "bright light"],
    y_var="mel_supp",
    x_order=["bright light"],
    x_colors=["gold"],
    x_var="condition",
    x_subgroup_var="Pupil state",
    x_subgroup_palette="muted",
    title="",
    x_label="",
    y_label="Melatonin suppression (%)",
    jitter=(-0.09, 0.15),
    spread_factor=0.05,
    offset=-0.5,
    box_width=0.32,
    violin_alpha=0.8,
    raw_marker_size=50,
)

# remove x ticks
# axes["B"].set_xticks([])
# axes["B"].set_xticklabels([])

# add more ticks to y
axes["B"].set_yticks(np.arange(-100, 101, 25))

# add grid
axes["B"].grid(
    True, which="both", axis="y", linestyle="--", linewidth=0.5, zorder=0
)
axes["B"].set_axisbelow(True)

# add title
axes["B"].set_title("Melatonin suppression", pad=10)

## Figure format
plt.tight_layout()
plt.show()

output_file = f"{figs_path}/mel_group_AUC_and_supp_VR_paper_light"
fig.savefig(f"{output_file}.png", dpi=500, bbox_inches="tight")
fig.savefig(f"{output_file}.svg", format="svg", bbox_inches="tight")
plt.close(fig)


# %% Calculate summary statistics

auc_df_2 = auc_df[auc_df["condition"] == "bright light"]


# Function to calculate custom statistics
def calculate_stats(series):
    return pd.Series(
        {
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "25%": series.quantile(0.25),
            "median": series.median(),
            "75%": series.quantile(0.75),
            "max": series.max(),
        }
    )


# Calculate stats for total
total_stats = calculate_stats(auc_df_2["mel_supp"]).to_frame().T
total_stats.index = ["Total"]

# Calculate stats for each subgroup
subgroup_stats = (
    auc_df_2.groupby("Pupil state")["mel_supp"]
    .apply(calculate_stats)
    .transpose()
)
