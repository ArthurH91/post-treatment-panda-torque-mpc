import pandas as pd
import bagpy
import pickle
import pathlib
import pinocchio as pin
import hppfcl as fcl

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as m_patches
from matplotlib.collections import PatchCollection
import seaborn as sns

from panda_loader import load_panda


DEFAULT_FONT_SIZE = 35
DEFAULT_AXIS_FONT_SIZE = DEFAULT_FONT_SIZE
DEFAULT_LINE_WIDTH = 4  # 13
DEFAULT_MARKER_SIZE = 4
DEFAULT_FONT_FAMILY = "sans-serif"
DEFAULT_FONT_SERIF = [
    "Times New Roman",
    "Times",
    "Bitstream Vera Serif",
    "DejaVu Serif",
    "New Century Schoolbook",
    "Century Schoolbook L",
    "Utopia",
    "ITC Bookman",
    "Bookman",
    "Nimbus Roman No9 L",
    "Palatino",
    "Charter",
    "serif",
]
DEFAULT_FIGURE_FACE_COLOR = "white"  # figure facecolor; 0.75 is scalar gray
DEFAULT_LEGEND_FONT_SIZE = 30  # DEFAULT_FONT_SIZE
DEFAULT_AXES_LABEL_SIZE = DEFAULT_FONT_SIZE  # fontsize of the x any y labels
DEFAULT_TEXT_USE_TEX = False
LINE_ALPHA = 0.9
SAVE_FIGURES = False
FILE_EXTENSIONS = ["pdf", "png"]  # ,'eps']
FIGURES_DPI = 150
SHOW_FIGURES = False
FIGURE_PATH = "./plot/"

mpl.rcdefaults()
mpl.rcParams["lines.linewidth"] = DEFAULT_LINE_WIDTH
mpl.rcParams["lines.markersize"] = DEFAULT_MARKER_SIZE
mpl.rcParams["patch.linewidth"] = 1
mpl.rcParams["font.family"] = DEFAULT_FONT_FAMILY
mpl.rcParams["font.size"] = DEFAULT_FONT_SIZE
mpl.rcParams["font.serif"] = DEFAULT_FONT_SERIF
mpl.rcParams["text.usetex"] = DEFAULT_TEXT_USE_TEX
mpl.rcParams["axes.labelsize"] = DEFAULT_AXES_LABEL_SIZE
mpl.rcParams["axes.grid"] = True
mpl.rcParams["legend.fontsize"] = DEFAULT_LEGEND_FONT_SIZE
# opacity of of legend frame
mpl.rcParams["legend.framealpha"] = 1.0
mpl.rcParams["figure.facecolor"] = DEFAULT_FIGURE_FACE_COLOR
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
scale = 1.0
mpl.rcParams["figure.figsize"] = 30 * scale, 10 * scale  # 23, 18  # 12, 9
line_styles = 10 * ["b", "g", "r", "c", "y", "k", "m"]



pkl_path = pathlib.Path().resolve().parent / "pickle"
b = {}
for file in pkl_path.glob("*.pkl"):
    with open(file, "rb") as handle:
        b[file.name] = pickle.load(handle)


rmodel, cmodel = load_panda()

def compute_minimal_distances_between_collision_pairs(q):
    # Creates data models
    rdata = rmodel.createData()
    cdata = cmodel.createData()

    # Updating the models
    pin.framesForwardKinematics(rmodel, rdata, q)
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)

    # HPPFCL Queries
    req = fcl.DistanceRequest()
    res = fcl.DistanceResult()

    # List storing all the distances, from the first collision pair to the last in the order of addition in the collision model
    list_dist = []
    # Going through all the collision pairs
    for col_pair in cmodel.collisionPairs:
        # Geometry objects ID in the collision model
        pair1_id = col_pair.first
        pair2_id = col_pair.second

        dist = fcl.distance(
            cmodel.geometryObjects[pair1_id].geometry,
            cdata.oMg[pair1_id],
            cmodel.geometryObjects[pair2_id].geometry,
            cdata.oMg[pair2_id],
            req,
            res,
        )
        list_dist.append(dist)
    return list_dist

df_curr = b["2024-01-23-15-37-12_5_col.pkl"]
q_arr = df_curr[3]["joint_state.position"]
dist_array = np.zeros((q_arr.shape[0], 4))
for i in range(dist_array.shape[0]):
    dist_array[i] = compute_minimal_distances_between_collision_pairs(
        np.array(eval(q_arr[i]))
    )

# Convert distance to dataframe
dist_df = pd.DataFrame(
    columns=["Time", "Right finger", "Left finger", "Link 7 part 1", "Link 7 part 2"]
)
dist_df["Time"] = df_curr[3]["Time"] - df_curr[3]["Time"][0]
dist_df["Right finger"] = dist_array[:, 0]
dist_df["Left finger"] = dist_array[:, 1]
dist_df["Link 7 part 1"] = dist_array[:, 2]
dist_df["Link 7 part 2"] = dist_array[:, 3]
dist_df["Lower bound"] = np.full((dist_array.shape[0]), 0.1)
time_df = df_curr[2]
time_df["Time"] -= time_df["Time"][0]
time_df["data.nsecs"] = time_df["data.nsecs"].copy() * 1e-6

# Pick moment in the time
t_min = 3.0
t_max = 13.0

dist_df_cropped = dist_df.loc[
    (dist_df["Time"] > t_min) & (dist_df["Time"] < t_max)
].reset_index()
time_df_cropped = time_df.loc[
    (time_df["Time"] > t_min) & (time_df["Time"] < t_max)
].reset_index()
dist_df_cropped["Time"] -= dist_df_cropped["Time"][0]
time_df_cropped["Time"] -= time_df_cropped["Time"][0]

fig, axs = plt.subplots(nrows=2, sharex="col", figsize=(10, 7))

sns.set_palette("colorblind")

collision_pairs = {
    "Right finger": "saddlebrown",
    "Left finger": "darkgreen",
    "Link 7 part 1": "darkmagenta",
    "Link 7 part 2": "darkcyan",
}

for cp, col in collision_pairs.items():
    sns.lineplot(
        dist_df_cropped,
        x="Time",
        y=cp,
        ax=axs[0],
        label=cp,
        linewidth=8.0,
    )

sns.lineplot(
    dist_df_cropped,
    x="Time",
    y="Lower bound",
    ax=axs[0],
    label="Lower bound",
    color="#000000",
    linewidth=8.0,
    linestyle="-.",
)
sns.lineplot(
    time_df_cropped,
    x="Time",
    y="data.nsecs",
    ax=axs[1],
    color=sns.color_palette("colorblind")[2],
    linewidth=8.0
)


# axs[0].set_xlim([0, dist_df_cropped.tail(1)["Time"].item()])
axs[0].set_ylim([0.09, 0.21])
axs[0].set_yticks([0.1, 0.15, 0.2])

# plt.legend()
axs[0].set_ylabel("Collision Distance (m)")
axs[1].set_ylabel("Computation Time (ms)")
axs[1].set_xlabel("Time (s)")

axs[0].legend(
    fancybox=True,
    # framealpha=0.0,
    loc="upper right",
    bbox_to_anchor=(1.0, 1.15),
    shadow=True,
    borderaxespad=0,
    ncol=5,
)

# axs[0].get_yaxis().set_label_coords(-0.07, 0.5)
# axs[1].get_yaxis().set_label_coords(-0.07, 0.5)

plt.tight_layout()
plt.savefig("distance_to_time.svg", format='svg', pad_inches=10.5)
plt.show()