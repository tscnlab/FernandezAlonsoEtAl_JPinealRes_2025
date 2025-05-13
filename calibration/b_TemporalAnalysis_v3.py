# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 17:49:10 2022

@author: malonso
"""

import glob
import pandas as pd
import numpy as np
import os
import seaborn as sns

from scipy.fft import rfft, rfftfreq
from matplotlib import pyplot as plt

# %% INPUT
root_folder = r"D:\NextCloud\BinIntMel\screen calibration\Temporal results"

folder = os.path.join(
    root_folder,
    "2022.09.06_18.22_gamma_1_200secs",
)

samplingTime_ms = 0.04  # miliseconds

gamma = 1

device = "pe01"

harmonicFs = [0.5, 1, 2, 3, 4]

# %% ANALYSIS

listOfFiles = glob.glob(folder + "/*.pkl")
fileNames = [os.path.basename(x) for x in listOfFiles]

samplingTime_s = samplingTime_ms / 1e3
samplingRate_hz = 1 / samplingTime_s


for fd in listOfFiles:

    # open pickle
    df = pd.read_pickle(fd)

    # to save fft results
    d_frequency = []
    d_samplingFreq_hz = []
    d_duration_s = []
    d_N = []

    d_chA_amp = []
    d_chB_amp = []

    d_chA_harm05f_amp = []  # at 0.5f, f, 2f, 3f, 4f
    d_chA_harm2f_amp = []
    d_chA_harm3f_amp = []
    d_chA_harm4f_amp = []

    d_chB_harm05f_amp = []  # at 0.5f, f, 2f, 3f, 4f
    d_chB_harm2f_amp = []
    d_chB_harm3f_amp = []
    d_chB_harm4f_amp = []

    for ii in range(0, len(df)):

        print(ii)

        # get variables
        channelA_mv = df.iloc[ii]["ChannelA"]
        channelB_mv = df.iloc[ii]["ChannelB"]

        frequency = df.iloc[ii]["Frequency"]

        if frequency != 0 and frequency < 1:

            flickerPeriod = 1 / frequency * 2

            xlim_freq_min = 0
            xlim_freq = 2 * frequency

        elif frequency != 0 and frequency >= 1:

            flickerPeriod = 1 / frequency * 5

            xlim_freq_min = frequency * 0.7
            xlim_freq = frequency * 1.35

        elif frequency == 0:

            flickerPeriod = 0.1

            xlim_freq_min = 0
            xlim_freq = 120

        if flickerPeriod > 200:

            flickerPeriod = 200

        trueDuration_s = len(channelA_mv) * samplingTime_s
        precision = str(1 / np.round(trueDuration_s))[::-1].find(
            "."
        )  # How many decimal places can we measure? Given by 1/duration
        print("precision: " + str(precision))

        # %% Frequency Analysis

        N = len(channelA_mv)
        peakThreshold = 6e8

        # Channel A fft
        channelA_yf = rfft(channelA_mv)
        channelA_xf = rfftfreq(N, 1 / samplingRate_hz)

        # find peaks
        idx_A = np.argwhere(np.abs(channelA_yf) > peakThreshold)
        channelA_peaks_hz = np.concatenate(channelA_xf[idx_A], axis=None)
        channelA_peaks_amp = np.concatenate(
            np.abs(channelA_yf[idx_A]), axis=None
        )

        # Channel B fft
        channelB_yf = rfft(channelB_mv)
        channelB_xf = rfftfreq(N, 1 / samplingRate_hz)

        # find peaks
        idx_B = np.argwhere(np.abs(channelB_yf) > peakThreshold)
        channelB_peaks_hz = np.concatenate(channelB_xf[idx_B], axis=None)
        channelB_peaks_amp = np.concatenate(
            np.abs(channelB_yf[idx_B]), axis=None
        )

        # find amplitude at HARMONICS
        chA_harms = np.array([])
        chB_harms = np.array([])

        for har in harmonicFs:

            # CHANNEL A
            idxA = np.argwhere(
                np.round(channelA_xf, precision) == frequency * har
            )

            if idxA.size != 0 and idxA.size > 2:
                idxA = idxA[0]

            elif idxA.size == 0:

                idxA = np.argmin(np.abs(channelA_xf - (frequency * har)))

                if idxA.size > 2:
                    idxA = idxA[0]

                print(
                    "ChA difference: "
                    + str(np.abs(channelA_xf[idxA] - (frequency * har)))
                )

            # add amplitude
            chA_harms = np.concatenate(
                (chA_harms, np.abs(channelA_yf[idxA])), axis=None
            )

            # CHANNEL B
            idxB = np.argwhere(
                np.round(channelB_xf, precision) == frequency * har
            )

            if idxB.size != 0 and idxB.size > 2:
                idxB = idxB[0]

            elif idxB.size == 0:  # find closest value

                idxB = np.argmin(np.abs(channelB_xf - (frequency * har)))

                if idxB.size > 2:
                    idxB = idxB[0]

                print(
                    "ChB difference: "
                    + str(np.abs(channelB_xf[idxB] - (frequency * har)))
                )

            # add amplitude
            chB_harms = np.concatenate(
                (chB_harms, np.abs(channelB_yf[idxB])), axis=None
            )

        ## Save FFT results
        d_frequency.append(frequency)
        d_samplingFreq_hz.append(samplingRate_hz)
        d_duration_s.append(trueDuration_s)
        d_N.append(N)

        d_chA_amp.append(chA_harms[1])
        d_chB_amp.append(chB_harms[1])

        d_chA_harm05f_amp.append(chA_harms[0])
        d_chA_harm2f_amp.append(chA_harms[2])
        d_chA_harm3f_amp.append(chA_harms[3])
        d_chA_harm4f_amp.append(chA_harms[4])

        d_chB_harm05f_amp.append(chB_harms[0])
        d_chB_harm2f_amp.append(chB_harms[2])
        d_chB_harm3f_amp.append(chB_harms[3])
        d_chB_harm4f_amp.append(chB_harms[4])

        # %% downsampling
        # if frequency == 0:
        #     downsample_factor = 2
        #     channelA_xf = channelA_xf[::downsample_factor]
        #     channelA_yf = np.abs(channelA_yf)[::downsample_factor]
        #     channelB_xf = channelB_xf[::downsample_factor]
        #     channelB_yf = np.abs(channelB_yf)[::downsample_factor]

        ylim_amp = (
            np.max(
                np.concatenate(
                    (channelB_peaks_amp[1:], channelA_peaks_amp[1:])
                )
            )
            + peakThreshold
        )

        # Remove irrelevant peaks
        if frequency == 0:
            max_peak = 100
        else:
            max_peak = 65

        maskA = (channelB_peaks_hz < max_peak) & (channelB_peaks_hz > 0)
        maskB = (channelA_peaks_hz < max_peak) & (channelA_peaks_hz > 0)

        channelB_peaks_hz = channelB_peaks_hz[maskB]
        channelB_peaks_amp = channelB_peaks_amp[maskB]
        channelA_peaks_hz = channelA_peaks_hz[maskA]
        channelA_peaks_amp = channelA_peaks_amp[maskA]

        # %% PLOT

        # Set the white theme for seaborn
        sns.set_theme(style="white", font_scale=1.1)

        if frequency == 0:

            fig = plt.figure(constrained_layout=True, figsize=(8, 3), dpi=600)
            axes = fig.subplot_mosaic([["A", "A", "B"]])
            fig.suptitle("Constant background light")

        else:
            fig = plt.figure(
                constrained_layout=True, figsize=(8, 5.7), dpi=600
            )
            fig.set_constrained_layout_pads(
                hspace=0.05, wspace=0.05
            )  # padding
            axes = fig.subplot_mosaic([["A", "A"], ["B", "C"]])
            fig.suptitle(f"Frequency: {frequency} Hz")

        # Wave Plot (Plot A)
        axes["A"].plot(
            np.linspace(0, trueDuration_s, len(channelA_mv)),
            np.array(channelA_mv) / 1000,
            "-b",
            label="Left eye",
        )
        axes["A"].plot(
            np.linspace(0, trueDuration_s, len(channelB_mv)),
            np.array(channelB_mv) / 1000,
            "-g",
            label="Right eye",
        )

        axes["A"].set_xlim([0 + 0.10, flickerPeriod + 0.10])

        axes["A"].set_xlabel("Time (s)")
        axes["A"].set_ylabel("Voltage (V)")
        axes["A"].set_title(" ")

        if frequency == 0:
            fig.legend(
                labels=["Left eye", "Right eye"],  # Add labels
                loc="upper center",
                bbox_to_anchor=(0.3, 0.94),
                ncol=2,
                frameon=False,
            )
        else:
            fig.legend(
                labels=["Left eye", "Right eye"],  # Add labels
                loc="upper center",
                bbox_to_anchor=(0.5, 0.97),
                ncol=2,
                frameon=False,
            )

        # ______________
        # Frequency Spectrum (Plot B)
        axes["B"].plot(
            channelA_xf,
            np.abs(channelA_yf),
            ".-b",
            linewidth=0.7,
            markersize=4,
        )
        axes["B"].plot(
            channelB_xf,
            np.abs(channelB_yf),
            ".-g",
            linewidth=0.7,
            markersize=4,
        )

        axes["B"].plot([0, xlim_freq], [peakThreshold, peakThreshold], "--r")

        axes["B"].plot(channelA_peaks_hz, channelA_peaks_amp, "*r")

        axes["B"].plot(channelB_peaks_hz, channelB_peaks_amp, "*r")
        axes["B"].annotate(
            f"{channelB_peaks_hz[0]:.2f} Hz",
            (
                channelB_peaks_hz[0],
                channelB_peaks_amp[0],
            ),  # Position of the text
            textcoords="offset points",
            xytext=(5, 5),  # Offset for better readability
            fontsize=13,
            color="black",
        )

        axes["B"].set_xlim([xlim_freq_min, xlim_freq])
        axes["B"].set_ylim(top=ylim_amp)
        axes["B"].set_xlabel("Frequency (Hz)")
        axes["B"].set_ylabel("Amplitude")
        axes["B"].set_title("Frequency domain")

        # ______________________
        # Harmonics Plot (Plot C)
        if frequency != 0:

            axes["C"].plot(
                [xx * frequency for xx in harmonicFs], chA_harms, ":*b"
            )
            axes["C"].plot(
                [xx * frequency for xx in harmonicFs], chB_harms, ":*g"
            )

            axes["C"].set_xlabel("Frequency (Hz)")
            axes["C"].set_ylabel("Amplitude")

            axes["C"].set_xlabel("Frequency (Hz)")
            axes["C"].set_title("Harmonics")

        # Save figure
        output_file = f"{root_folder}/figures/{device}_{frequency}Hz"
        fig.savefig(f"{output_file}.png", dpi=600)
        if frequency == 5:
            fig.savefig(f"{output_file}.svg", format="svg")
        plt.show()

    # %% Save FFT results
    fftDF = pd.DataFrame(
        {
            "Frequency_hz": d_frequency,
            "SamplingRate_hz": d_samplingFreq_hz,
            "Duration_s": d_duration_s,
            "TotalSamples": d_N,
            "ChannelA_yAmp": d_chA_amp,
            "ChannelB_yAmp": d_chB_amp,
            "ChA_05f_amp": d_chA_harm05f_amp,
            "ChA_2f_amp": d_chA_harm2f_amp,
            "ChA_3f_amp": d_chA_harm3f_amp,
            "ChA_4f_amp": d_chA_harm4f_amp,
            "ChB_05f_amp": d_chB_harm05f_amp,
            "ChB_2f_amp": d_chB_harm2f_amp,
            "ChB_3f_amp": d_chB_harm3f_amp,
            "ChB_4f_amp": d_chB_harm4f_amp,
        }
    )

    fftDF.to_csv(
        f"{root_folder}/ResultsFFT_.csv",
        sep=",",
        index=False,
    )

print("Done!")
