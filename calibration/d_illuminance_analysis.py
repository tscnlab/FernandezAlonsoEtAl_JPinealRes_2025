import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import glob
import os
from matplotlib.lines import Line2D

# %% FUNCTIONS


def load_file(file_path):
    if file_path.endswith(".pkl"):
        with open(file_path, "rb") as f:
            df = pickle.load(f)
    else:
        return None

    if "VR_number" not in df.columns:
        df.insert(1, "VR_number", 2)

    return df


# %% LOAD DATA

data_folder = r"D:\NextCloud\BinIntMel\code\BinIntMel-screenCali\measure-illuminance\results"
figs_path = r"D:\NextCloud\BinIntMel\reports\paper  - VR method\1_First round revisions\figures"

file_paths = glob.glob(os.path.join(data_folder, "*.pkl"))
dataframes = [load_file(fp) for fp in file_paths if load_file(fp) is not None]
df = pd.concat(dataframes, ignore_index=True)

# %% METRICS

metrics = [
    ("illuminance_lux", "Photopic illuminance", "Illuminance (lux)"),
    ("a_opic_iprgc", "Melanopic EDI", r"Melanopic $E_{D65}$ (lx)"),
    ("a_opic_rods", "Rhodopic EDI", r"Rod $E_{D65}$ (lx)"),
    ("a_opic_l_cone", "L-cone-opic EDI", r"L-cone $E_{D65}$ (lx)"),
    ("a_m_cone", "M-cone-opic EDI", r"M-cone $E_{D65}$ (lx)"),
    ("a_s_cone", "S-cone-opic EDI", r"S-cone $E_{D65}$ (lx)"),
]

vr_colors = sns.color_palette("tab10", n_colors=5)

# %% PLOT PER EYE

for eye in ["left", "right"]:

    fig, axes = plt.subplots(2, 3, figsize=(8, 5), sharex=True)
    axes = axes.flatten()

    for idx, (metric, short_title, y_label) in enumerate(metrics):
        ax = axes[idx]
        eye_df = df[df["eye"] == eye]

        # Plot each VR headset line
        group = (
            eye_df.groupby(["VR_number", "input_intensity"])[metric]
            .mean()
            .reset_index()
        )
        for vr, subdata in group.groupby("VR_number"):
            color = vr_colors[int(vr) % len(vr_colors)]
            ax.plot(
                subdata["input_intensity"],
                subdata[metric],
                linestyle="-",
                color=color,
                label=f"VR{int(vr):02d}",
            )

        # Mean ± SD line
        mean_std = (
            eye_df.groupby("input_intensity")[metric]
            .agg(["mean", "std"])
            .reset_index()
        )
        ax.errorbar(
            mean_std["input_intensity"],
            mean_std["mean"],
            yerr=mean_std["std"],
            fmt=".",
            color="black",
            linewidth=1.5,
            capsize=3,
            elinewidth=1.5,
            label=None,
        )

        ax.set_title(short_title)
        if idx > 2:
            ax.set_xlabel("Input Intensity")
        ax.set_ylabel(y_label)
        ax.set_ylim(0, 200)
        ax.set_xlim(0, 1)

    # Add custom "Mean ± SD" as FIRST item in legend
    handles, labels = axes[0].get_legend_handles_labels()

    mean_sd_line = Line2D(
        [],
        [],
        color="black",
        marker="o",
        linestyle="-",  # short horizontal line
        linewidth=2,
        markersize=6,
        label="Mean ± SD",
    )

    handles.insert(0, mean_sd_line)
    labels.insert(0, "Mean ± SD")

    # Shared legend at bottom
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=6,
        fontsize="small",
    )
    legend.get_frame().set_visible(False)

    fig.suptitle(f"{eye.capitalize()} eye")
    fig.tight_layout(rect=[0, 0.05, 1, 1.01])

    plt.show()

    # Save
    output_file = f"{figs_path}/irradiance_measurements_{eye}"

    fig.savefig(f"{output_file}.png", dpi=500)
    fig.savefig(f"{output_file}.svg", format="svg")

    plt.close(fig)

# %% Generate Table

import pandas as pd

# Metrics to include
metrics = {
    "illuminance_lux": "Illuminance (lux)",
    "a_opic_iprgc": "Melanopic ED65 (lx)",
    "a_opic_rods": "Rod ED65 (lx)",
    "a_opic_l_cone": "L-cone ED65 (lx)",
    "a_m_cone": "M-cone ED65 (lx)",
    "a_s_cone": "S-cone ED65 (lx)",
}


# Group by input intensity and calculate mean and std
summary = (
    df.groupby("input_intensity")[list(metrics.keys())]
    .agg(["mean", "std"])
    .reset_index()
)

# Build formatted table: mean ± std
formatted_data = {"Input Intensity": summary["input_intensity"]}

for metric_key, label in metrics.items():
    mean = summary[(metric_key, "mean")]
    std = summary[(metric_key, "std")]
    formatted_data[label] = [f"{m:.3f} ± {s:.1f}" for m, s in zip(mean, std)]

# Create dataframe
formatted_df = pd.DataFrame(formatted_data)


formatted_df.to_excel(f"{figs_path}/summary_table_mean_std.xlsx", index=False)

# %% Spectral irradiance
import numpy as np

sns.set_theme(font_scale=1.1, style="white")

wavelengths = np.arange(380, 781, 1)
spectral_irradiance = df["spectrum"][10]

fig = plt.figure(figsize=(5, 5))
plt.plot(wavelengths, spectral_irradiance, linewidth=2)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Spectral Irradiance (W/m²/nm)")
plt.title("Spectral Irradiance")
plt.xlim(400, 750)
plt.ylim(0, None)
plt.tight_layout()
plt.show()

output_file = f"{figs_path}/spectral_irradiance"

fig.savefig(f"{output_file}.png", dpi=500)
fig.savefig(f"{output_file}.svg", format="svg")
