import os
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from datetime import datetime
from scipy.optimize import curve_fit


# %% FUNCTIONS
def get_unique_non_zero(rgb_list):
    unique_non_zero = list(set(rgb_list) - {0})
    if unique_non_zero:
        return unique_non_zero[0]
    else:
        return 0


def info_file_importer(filename, data_lines=(1, None)):
    """
    Imports data from an info file, ensuring datetime consistency using pandas' parse_dates.
    """
    # Define expected column names
    expected_columns = [
        "Eye",
        "Repetition",
        "Colour",
        "RGB",
        "PCStartDatetime",
        "PCEndDatetime",
        "Outcome",
    ]

    # Read the data using pandas, parsing datetime columns directly
    df = pd.read_csv(
        filename,
        skiprows=data_lines[0] - 1,  # Adjust for Python indexing
        nrows=(
            None
            if data_lines[1] is None
            else data_lines[1] - data_lines[0] + 1
        ),
        names=expected_columns,
        header=0,  # Use the first row as column headers
    )

    # Ensure datetime columns are parsed with the correct format
    date_format = "%Y-%m-%d %H:%M:%S"
    df["PCStartDatetime"] = pd.to_datetime(
        df["PCStartDatetime"], format=date_format, errors="coerce"
    )
    df["PCEndDatetime"] = pd.to_datetime(
        df["PCEndDatetime"], format=date_format, errors="coerce"
    )

    # Convert 'RGB' column to numerical arrays
    def process_rgb(rgb_str):
        try:
            return [float(num) for num in rgb_str.strip("[]").split()]
        except Exception:
            return np.nan

    df["rgb"] = df["RGB"].apply(process_rgb)
    df = df.drop(columns=["RGB"])  # Drop the original RGB column

    return df


def jeti_importer(filepath):
    try:
        # Read spectral radiance data
        spectral_radiance = pd.read_csv(
            filepath, skiprows=53, names=["Wave", "Radiance"]
        )
        # Read info data
        info = pd.read_csv(filepath, nrows=20, names=["Variable", "Value"])

        # Helper function to find the value by variable name
        def find_value(var_name):
            match = info.loc[info["Variable"].str.strip() == var_name, "Value"]
            if not match.empty:
                return match.values[0]
            else:
                raise ValueError(
                    f"Variable '{var_name}' not found in the info data."
                )

        # Extract necessary values
        date = find_value("Date")
        time = find_value("Time")
        dtime = datetime.strptime(date + " " + time, "%m/%d/%Y %I:%M:%S%p")

        result = {
            "Wave": spectral_radiance["Wave"].to_list(),
            "Radiance": spectral_radiance["Radiance"].to_list(),
            "Date": date,
            "Time": time,
            "IntegrationTime_ms": float(find_value("Integration Time [ms]")),
            "LuminanceFromDevice": float(find_value("Luminance [cd/sqm]")),
            "TotalRadianceFromDevice": float(
                find_value("Radiance [W/sr*sqm]")
            ),
            "DateTime": dtime,
            "FileName": os.path.basename(filepath),
        }
        return result

    except Exception as e:
        if "info" in locals():
            print(info)
        print(f"Error processing file {filepath}: {e}")
        return None


def import_and_merge_data(results_dir):

    # Import info files
    info_files = [
        f
        for f in os.listdir(results_dir)
        if f.startswith("Radiance") and f.endswith(".csv")
    ]
    in_data = pd.DataFrame()

    for file in info_files:
        file_data = info_file_importer(os.path.join(results_dir, file))
        in_data = pd.concat([in_data, file_data], ignore_index=True)

    # Import Jeti results
    jeti_dir = os.path.join(results_dir, "jeti")
    jeti_files = [
        f
        for f in os.listdir(jeti_dir)
        if f.startswith("202") and f.endswith(".csv")
    ]
    jeti_data = pd.DataFrame()

    for file in jeti_files:
        file_data = jeti_importer(os.path.join(jeti_dir, file))
        jeti_data = pd.concat(
            [jeti_data, pd.DataFrame([file_data])], ignore_index=True
        )

    # Merge corresponding rows and calculate averages
    merged_data = in_data.copy()
    merged_data["IntegrationTime_ms"] = np.nan
    merged_data["LuminanceFromDevice"] = np.nan
    merged_data["TotalRadianceFromDevice"] = np.nan

    # Keep track of unassigned rows during iteration
    # unassigned_files = set(jeti_data["FileName"].dropna())

    for idx, row in merged_data.iterrows():
        relevant_rows = jeti_data[
            (jeti_data["DateTime"] >= row["PCStartDatetime"])
            & (jeti_data["DateTime"] < row["PCEndDatetime"])
        ]

        if not relevant_rows.empty:
            last_row = relevant_rows.iloc[-1]
            merged_data.loc[
                idx,
                [
                    "IntegrationTime_ms",
                    "LuminanceFromDevice",
                    "TotalRadianceFromDevice",
                    "FileName",
                ],
            ] = last_row[
                [
                    "IntegrationTime_ms",
                    "LuminanceFromDevice",
                    "TotalRadianceFromDevice",
                    "FileName",
                ]
            ]
            # unassigned_files.discard(last_row["FileName"])
        else:
            print(
                f"No data found from {row['PCStartDatetime']} to {row['PCEndDatetime']}"
            )

    # determine unasgined jeti files
    # if unassigned_files:
    #     print("Unassigned Jeti files:")
    #     for file in sorted(unassigned_files):
    #         print(file)

    merged_data = merged_data[~merged_data["LuminanceFromDevice"].isna()]

    merged_data["input"] = merged_data["rgb"].apply(get_unique_non_zero)

    return merged_data


# %% PLOTS


def gamma_function(x, a, gamma):
    return a * x**gamma


def linear_function(x, a, b):
    return a * x + b


def plot_data_fit_gamma(
    merged_data: pd.DataFrame,
    ax=None,
    fig_title="",
    y_label="Luminance [cd/m$^2$]",
    colors: dict = {"w": "grey", "r": "red", "g": "green", "b": "blue"},
):

    x_plot = np.linspace(0.1, 1, 100)

    # Plot raw data and gamma fits

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    for color, plot_color in colors.items():
        x_data = merged_data.loc[merged_data["Colour"] == color, "rgb"].apply(
            get_unique_non_zero
        )
        y_data = merged_data.loc[
            merged_data["Colour"] == color, "LuminanceFromDevice"
        ]

        # Calculate mean and standard deviation for error bars
        grouped_data = (
            pd.DataFrame({"x": x_data, "y": y_data})
            .groupby("x")
            .agg(["mean", "std"])
            .reset_index()
        )
        x_clean = grouped_data["x"]
        y_mean = grouped_data[("y", "mean")]
        y_std = grouped_data[("y", "std")]

        # Plot error bars
        legend_label = (
            "white (mean ± SD)"
            if color == "w"
            else f"{plot_color} (mean ± SD)"
        )
        ax.errorbar(
            x_clean,
            y_mean,
            yerr=y_std,
            fmt="o",
            color=plot_color,
            ecolor=plot_color,
            elinewidth=1,
            capsize=3,
            label=legend_label,
        )

        if not x_data.empty:
            # Group and average data to handle duplicates
            grouped_data = (
                pd.DataFrame({"x": x_data, "y": y_data})
                .groupby("x")
                .mean()
                .reset_index()
            )
            x_clean = grouped_data["x"].values
            y_clean = grouped_data["y"].values

            # Fit the gamma function with both 'a' and 'gamma'
            try:
                params, _ = curve_fit(
                    gamma_function, x_clean, y_clean, p0=[1.0, 2.2]
                )
                a, gamma = params
                # Plot the gamma curve
                ax.plot(
                    x_plot,
                    gamma_function(x_plot, a, gamma),
                    color=plot_color,
                    label=f"{legend_label.split()[0]} fit (γ={gamma:.2f})",
                )
            except RuntimeError:
                print(f"Fitting failed for color {plot_color}")
                continue

    # Customize legend
    handles, labels = ax.get_legend_handles_labels()

    # Move error bars to the top of the legend
    sorted_handles_labels = sorted(
        zip(handles, labels), key=lambda hl: "mean ± SD" in hl[1], reverse=True
    )
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)

    ax.legend(sorted_handles, sorted_labels)
    ax.set_xlabel("RGB Input")
    ax.set_ylabel(y_label)
    ax.set_title(fig_title)

    return fig, ax


def plot_fit_linear(
    merged_data: pd.DataFrame,
    ax=None,
    fig_title="",
    y_label="Luminance [cd/m$^2$]",
    colors: dict = {"w": "grey", "r": "red", "g": "green", "b": "blue"},
):

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    for color, plot_color in colors.items():
        x_data = merged_data.loc[merged_data["Colour"] == color, "rgb"].apply(
            get_unique_non_zero
        )
        y_data = merged_data.loc[
            merged_data["Colour"] == color, "LuminanceFromDevice"
        ]

        # Calculate mean and standard deviation for error bars
        grouped_data = (
            pd.DataFrame({"x": x_data, "y": y_data})
            .groupby("x")
            .agg(["mean", "std"])
            .reset_index()
        )
        x_clean = grouped_data["x"]
        y_mean = grouped_data[("y", "mean")]
        y_std = grouped_data[("y", "std")]

        # Plot error bars
        legend_label = (
            "white (mean ± SD)"
            if color == "w"
            else f"{plot_color} (mean ± SD)"
        )
        ax.errorbar(
            x_clean,
            y_mean,
            yerr=y_std,
            fmt="o",
            color=plot_color,
            ecolor=plot_color,
            elinewidth=1,
            capsize=3,
            label=legend_label,
        )

        # plot linear fit
        if not x_data.empty:
            # Group and average data to handle duplicates
            grouped_data = (
                pd.DataFrame({"x": x_data, "y": y_data})
                .groupby("x")
                .mean()
                .reset_index()
            )
            x_clean = grouped_data["x"].values
            y_clean = grouped_data["y"].values

            # Fit the linear function
            try:
                params, _ = curve_fit(linear_function, x_clean, y_clean)
                a, b = params

                # Calculate predicted values and R-squared
                y_pred = linear_function(x_clean, a, b)
                ss_res = np.sum((y_clean - y_pred) ** 2)
                ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                # Plot the linear fit
                ax.plot(
                    x_clean,
                    y_pred,
                    color=plot_color,
                    label=f"{legend_label.split()[0]} linear fit (R²={r_squared:.1f})",
                )
            except RuntimeError:
                print(f"Fitting failed for color {plot_color}")

    # Customize legend
    handles, labels = ax.get_legend_handles_labels()

    # Move error bars to the top of the legend
    sorted_handles_labels = sorted(
        zip(handles, labels), key=lambda hl: "mean ± SD" in hl[1], reverse=True
    )
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)

    ax.legend(sorted_handles, sorted_labels)
    ax.set_xlabel("RGB Input")
    ax.set_ylabel(y_label)
    ax.set_title(fig_title)

    return fig, ax


def plot_raw_data(
    merged_data: pd.DataFrame,
    ax=None,
    fig_title="",
    colors: dict = {"w": "grey", "r": "red", "g": "green", "b": "blue"},
):

    if ax is None:
        fig, ax = plt.figure(figsize=(8, 6))
    else:
        fig = ax.figure

    for color, plot_color in colors.items():
        x_data = merged_data.loc[merged_data["Colour"] == color, "rgb"].apply(
            get_unique_non_zero
        )
        y_data = merged_data.loc[
            merged_data["Colour"] == color, "LuminanceFromDevice"
        ]

        if not x_data.empty:
            # Group and calculate mean and standard deviation
            grouped_data = (
                pd.DataFrame({"x": x_data, "y": y_data})
                .groupby("x")
                .agg(["mean", "std"])
                .reset_index()
            )
            x_clean = grouped_data["x"]
            y_mean = grouped_data[("y", "mean")]
            y_std = grouped_data[("y", "std")]

            # Plot mean points with error bars
            legend_label = (
                "white mean ± SD"
                if color == "w"
                else f"{plot_color} (mean ± SD)"
            )
            plt.errorbar(
                x_clean,
                y_mean,
                yerr=y_std,
                fmt="o",
                color=plot_color,
                ecolor=plot_color,
                elinewidth=1,
                capsize=3,
                label=legend_label,
            )

    plt.xlabel("RGB Input")
    plt.ylabel("Luminance [cd/m$^2$]")
    plt.title(fig_title)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return fig, ax


# %% Set figures style
sns.set_theme(font_scale=1.1, style="white")

# %% RUN: raw data and gamma fits
root_dir = r"D:\NextCloud\BinIntMel\screen calibration\Spectral results"
folders_dict = {
    "Left eye": "22.08.11_pe-01_left eye_try2",
    "Right eye": "22.07.20_pe-01_right eye",
}

figs, axs = plt.subplots(1, 2, figsize=(8.5, 6))
y_min, y_max = 0, float("-inf")

for idx, (eye, folder) in enumerate(folders_dict.items()):

    # Define files path
    results_dir = os.path.join(root_dir, folder)

    # Import data
    merged_data = import_and_merge_data(results_dir=results_dir)

    # Plot
    plot_data_fit_gamma(
        merged_data=merged_data,
        ax=axs[idx],
        fig_title=eye,
        y_label="Luminance [cd/m$^2$]" if idx == 0 else "",
    )

    # Get current axis limits and update overall min and max
    current_y_min, current_y_max = axs[idx].get_ylim()
    y_min = min(y_min, current_y_min)
    y_max = max(y_max, current_y_max)

    axs[idx].grid(False)

for ax in axs:
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, ax.get_xlim()[1])

plt.tight_layout()
plt.show()

figs.savefig(
    f"{root_dir}/data_and_gamma_fit_left_right.png",
    dpi=600,
    bbox_inches="tight",
)
figs.savefig(f"{root_dir}/data_and_gamma_fit_left_right.svg", format="svg")
plt.close(figs)

# %% RUN: comfirm if linear

results_dir = r"D:\NextCloud\BinIntMel\screen calibration\Spectral results\22.08.11_pe-01_linear_gamma_2.2252_try2"

merged_data = import_and_merge_data(results_dir=results_dir)

fig, ax = plt.subplots(figsize=(4.5, 4.5))

plot_fit_linear(
    merged_data=merged_data,
    fig_title="Gamma-corrected display (γ=2.225)",
    y_label="Luminance [cd/m$^2$]",
    colors={"w": "grey"},
    ax=ax,
)
ax.set_xlim(0, ax.get_xlim()[1])
ax.set_ylim(0, ax.get_ylim()[1])

plt.tight_layout()
plt.show()

fig.savefig(
    f"{root_dir}/gamma_corrected_right_eye.png", dpi=600, bbox_inches="tight"
)
fig.savefig(f"{root_dir}/gamma_corrected_right_eye.svg", format="svg")

plt.close(fig)
