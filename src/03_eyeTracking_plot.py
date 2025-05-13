import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
file_dir = (
    r"D:\NextCloud\BinIntMel\data_main\derivatives\eyeTrack_2025032.parquet"
)

df = pd.read_parquet(file_dir)


# %%
def sem(x):
    return np.nanstd(x, ddof=1) / np.sqrt(np.count_nonzero(~np.isnan(x)))


# Step 1: Filter conditions
filtered_df = df[df["condition"].isin(["constant", "dark"])]

# %%


# Step 2: Group by participant and condition
summary = (
    filtered_df.groupby(["participantID", "condition"])
    .agg(
        {
            "Left pupil diam": ["mean", "median", sem],
            "Right pupil diam": ["mean", "median", sem],
        }
    )
    .reset_index()
)

# Flatten multi-level column headers into a single line
summary.columns = [
    f"{col[0]} {col[1]}" if col[1] else col[0] for col in summary.columns
]

summary = summary.rename(
    columns={
        "Left pupil diam mean": "Left pupil mean",
        "Left pupil diam median": "Left pupil median",
        "Left pupil diam <lambda_0>": "Left pupil SEM",
        "Right pupil diam mean": "Right pupil mean",
        "Right pupil diam median": "Right pupil median",
        "Right pupil diam <lambda_0>": "Right pupil SEM",
    }
)

print(
    f"{summary['Left pupil mean'].isna().sum()} NA samples out of {len(summary)}"
)

# %%


summary = summary.dropna(
    subset=["Left pupil mean", "Right pupil mean"], how="all"
)


summary["Pupil state"] = np.where(
    summary["participantID"].astype(int) < 150, "natural", "dilated"
)

# Compute the average pupil size per row
summary["Mean pupil size"] = summary[
    ["Left pupil mean", "Right pupil mean"]
].mean(axis=1)

# %%
# Plot connected boxplot

sns.set_theme(font_scale=1.1, style="white")


fig = plt.figure(figsize=(5, 4))

# Boxplot with custom fill colors
box = sns.boxplot(
    data=summary,
    x="condition",
    y="Mean pupil size",
    width=0.5,
    showfliers=False,
    palette={"dark": "darkgray", "constant": "gold"},
    linewidth=2,
)

subgroup_palette = {
    "natural": sns.color_palette("muted")[0],
    "dilated": sns.color_palette("muted")[1],
}

# Add individual participant data points
sns.stripplot(
    data=summary,
    x="condition",
    y="Mean pupil size",
    hue="Pupil state",
    size=6,
    jitter=True,
    alpha=0.8,
    dodge=False,
    palette=subgroup_palette,
    edgecolor="black",
    linewidth=0.7,
)

# Update tick labels AFTER plotting
# plt.xticks(ticks=[0, 1], labels=["dark", "bright light"])

# Labels & layout
plt.title("Average pupil size for both eyes")
plt.ylabel("Pupil diameter (mm)")
plt.xlabel("Condition")
plt.legend(title="Pupil state", loc="lower left")

plt.tight_layout()
plt.show()

# %%
figs_path = r"D:\NextCloud\BinIntMel\reports\paper  - VR method\1_First round revisions\figures"

output_file = f"{figs_path}/mel_group_AUC_and_supp_VR_paper_light"
fig.savefig(f"{output_file}.png", dpi=500, bbox_inches="tight")
fig.savefig(f"{output_file}.svg", format="svg", bbox_inches="tight")
