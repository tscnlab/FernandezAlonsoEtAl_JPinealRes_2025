# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 17:49:10 2022

@author: malonso
"""
import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# ______________________________________________
# INPUT

mainPath = r"D:\NextCloud\BinIntMel\screen calibration\Temporal results"
file = os.path.join(mainPath, "ResultsFFT_Temporal_Combined.csv")

harmonicFs = [0.5, 1, 2, 3, 4]


# open pickle
df = pd.read_csv(file)


# %% Amplitude at f PLOT
sns.set_theme(style="white", font_scale=1.2)

yticks = [x * 1e9 for x in [0, 0.5, 1, 1.5, 2.0]]

fig = plt.figure(constrained_layout=True, figsize=(7, 3.5), dpi=600)
axes = fig.subplot_mosaic("AB")


axes["A"].plot(
    df.Frequency_hz,
    df.ChannelA_yAmp,
    "--*b",
    label="Left eye",
    linewidth=0.7,
    markersize=4,
)
axes["A"].plot(
    df.Frequency_hz,
    df.ChannelB_yAmp,
    "--*g",
    label="Right eye",
    linewidth=0.7,
    markersize=4,
)

axes["A"].set_yticks(yticks)
axes["A"].set_xlabel("Frequency (Hz)")
axes["A"].set_ylabel("Amplitude")
axes["A"].legend(loc="lower left")


# axes["A"].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
# ncol=2, mode="expand", borderaxespad=0.)

axes["A"].set_title("Temporal transfer function")


# Harmonics
# channel A
for row in range(len(df)):

    freq = df.Frequency_hz[row]
    xf = [0.5, 1, 2, 3, 4]
    harmChA = [
        df.ChA_05f_amp[row],
        df.ChannelA_yAmp[row],
        df.ChA_2f_amp[row],
        df.ChA_3f_amp[row],
        df.ChA_4f_amp[row],
    ]
    # harmChB = [df.ChB_05f_amp[row], df.ChannelB_yAmp[row], df.ChB_2f_amp[row], df.ChB_3f_amp[row], df.ChB_4f_amp[row]]

    axes["B"].plot(
        xf, harmChA, ".-", linewidth=0.7, markersize=4, label=str(freq) + " Hz"
    )

    # axes["B"].plot(xf,harmChB,
    #                '.-', linewidth=0.7, markersize=4, label = str(freq) + ' Hz')


# format
axes["B"].set_yticks(yticks)
axes["B"].set_xlabel("xf")
# axes["B"].set_ylabel("Amplitude")
axes["B"].legend(loc="upper right", fontsize=10)
axes["B"].set_title("Amplitude at harmonics", pad=19)


# save
output_file = f"{mainPath}/figures/TemporalTransferFunction"
fig.savefig(f"{output_file}.png", dpi=600)
fig.savefig(f"{output_file}.svg", format="svg")


# %% Temporal transfer at harmonics

yLim_max = 1e8

fig = plt.figure(constrained_layout=True, figsize=(8, 6))
axes = fig.subplot_mosaic([["A", "B"], ["C", "D"]])

# 0.5f
axes["A"].plot(
    df.Frequency_hz,
    df.ChA_05f_amp,
    ":*b",
    label="Channel A",
    linewidth=0.7,
    markersize=4,
)
axes["A"].plot(
    df.Frequency_hz,
    df.ChB_05f_amp,
    ":*g",
    label="Channel B",
    linewidth=0.7,
    markersize=4,
)


axes["A"].set_xlabel("Frequency (Hz)")
axes["A"].set_ylabel("Amplitude")
axes["A"].legend(loc="upper left")
axes["A"].set_title("0.5f")
axes["A"].set_ylim([0, yLim_max])


# 2f
axes["B"].plot(
    df.Frequency_hz,
    df.ChA_2f_amp,
    ":*b",
    label="Channel A",
    linewidth=0.7,
    markersize=4,
)
axes["B"].plot(
    df.Frequency_hz,
    df.ChB_2f_amp,
    ":*g",
    label="Channel B",
    linewidth=0.7,
    markersize=4,
)

axes["B"].set_xlabel("Frequency (Hz)")
axes["B"].set_title("2f")
axes["B"].set_ylim([0, yLim_max])


# 3f
axes["C"].plot(
    df.Frequency_hz,
    df.ChA_3f_amp,
    ":*b",
    label="Channel A",
    linewidth=0.7,
    markersize=4,
)
axes["C"].plot(
    df.Frequency_hz,
    df.ChB_3f_amp,
    ":*g",
    label="Channel B",
    linewidth=0.7,
    markersize=4,
)

axes["C"].set_xlabel("Frequency (Hz)")
axes["C"].set_title("3f")
axes["C"].set_ylim([0, yLim_max])


# 4f
axes["D"].plot(
    df.Frequency_hz,
    df.ChA_4f_amp,
    ":*b",
    label="Channel A",
    linewidth=0.7,
    markersize=4,
)
axes["D"].plot(
    df.Frequency_hz,
    df.ChB_4f_amp,
    ":*g",
    label="Channel B",
    linewidth=0.7,
    markersize=4,
)

axes["D"].set_xlabel("Frequency (Hz)")
axes["D"].set_title("4f")
axes["D"].set_ylim([0, yLim_max])


# save
output_file = f"{mainPath}/figures/harmonics"
# fig.savefig(f"{output_file}.png", dpi=600)
# fig.savefig(f"{output_file}.svg", format="svg")


print("Done!")
