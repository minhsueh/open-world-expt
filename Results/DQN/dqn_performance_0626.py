import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator, MultipleLocator
import collections
import scipy
import os
import matplotlib.tri as tri
import numpy as np
import mpltern

print(os.getcwd())

# OUTPUT_FOLDER = "0517_dqn_agent_testing/"
OUTPUT_FOLDER = "0626_dqn_agent_testing/"
model_name = "0517_r0_d0_p9_T300_replay_frame_lr0.0001_delta0.9"
novelty_list = ["NN"]
OUTPUT_FILE_PATH = f"./agents/results/{OUTPUT_FOLDER}"


def errorbar_plot(model_name, nov, measurement, output_path):

    file_name = f"{model_name}_{nov}.csv"
    output_file_name = file_name.split(".csv")[0]

    # df = pd.read_csv(OUTPUT_FILE_PATH + nov + "/" + model_name + "/" + file_name)
    df = pd.read_csv(OUTPUT_FILE_PATH + "0517_r0_d0_p9_T300_replay_frame_lr0.0001_delta0.9_NN_seed20.csv")
    # df.to_csv(output_path + file_name, index=False)

    # plotting
    color_list = ["red", "green", "darkviolet", "blue"]
    model_list = ["player_1", "agent_random", "agent_dump", "agent_p"]
    label_map = {"player_1": "AgentDQN", "agent_random": "AgentR", "agent_dump": "AgentD", "agent_p": "AgentP"}
    ylabel_map = {"cash": "Cash", "win_rate": "Winning rate", "rank": "Rank", "winning_percentage": "Winning percentage"}
    figure, ax = plt.subplots()
    for model, color in zip(model_list, color_list):
        if model == "player_1":
            markers, caps, bars = ax.errorbar(
                df["backgound_agent_list"],
                df[model + "_" + measurement],
                df[model + "_" + measurement + "_ste"],
                c=color,
                label=label_map[model],
                capsize=5,
                alpha=0.6,
                fmt="x",
            )
            [bar.set_alpha(0.3) for bar in bars]
            [cap.set_alpha(0.3) for cap in caps]
            """
            ax.plot(
                df["backgound_agent_list"],
                df[model + "_" + measurement],
                c=color,
                label=label_map[model] + "(" + df["model"].to_list()[0] + ")",
                marker="x",
                ls="None",
            )
            """
        elif not df[model + "_cash"].isnull().values.all():
            markers, caps, bars = ax.errorbar(
                df["backgound_agent_list"],
                df[model + "_" + measurement],
                df[model + "_" + measurement + "_ste"],
                c=color,
                label=label_map[model],
                capsize=5,
                alpha=0.6,
                fmt="o",
            )
            [bar.set_alpha(0.3) for bar in bars]
            [cap.set_alpha(0.3) for cap in caps]

            # significant
            significant_list = df[measurement + "_significant_than_" + model].to_list()
            x_height_list = df[model + "_" + measurement].to_list()
            x_list = df["backgound_agent_list"].to_list()
            if measurement == "rank":
                scaling_factor = 1.005
            else:
                scaling_factor = 1.1
            for i in range(len(significant_list)):
                if significant_list[i] == "*":
                    plt.text(x_list[i], x_height_list[i] * scaling_factor, significant_list[i], c=color)

    plt.grid()
    # plt.xticks(range(1,max_game_num+1))
    plt.xlabel("Background agents")
    plt.ylabel(ylabel_map[measurement])
    plt.xticks(rotation=90)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    ax.figure.savefig(output_path + output_file_name + "_avg_" + measurement + ".png")
    plt.clf()
    plt.close()


def tri_plot(file_name, measurement, output_path):
    color_map = {"r": "green", "d": "blue", "p": "darkviolet"}
    axis_map = {"t": "r", "l": "d", "r": "p"}
    clim_map = {"cash": (0, 2000), "winning_percentage": (0, 100), "win_rate": (0, 1), "rank": (0, 10)}

    output_file_name = file_name.split(".csv")[0]

    df = pd.read_csv(OUTPUT_FILE_PATH + file_name)
    df.to_csv(output_path + file_name, index=False)

    df[["r", "d", "p"]] = df["backgound_agent_list"].str.split("_", expand=True)
    r = (df["r"].str[1]).astype(int).values
    d = (df["d"].str[1]).astype(int).values
    p = (df["p"].str[1]).astype(int).values

    value = df["player_1_" + measurement]

    fig = plt.figure()
    ax = fig.add_subplot(projection="ternary")

    # grid
    ax.grid()
    # Color ticks, grids, tick-labels
    ax.taxis.set_major_locator(MultipleLocator(0.1))
    ax.laxis.set_major_locator(MultipleLocator(0.1))
    ax.raxis.set_major_locator(MultipleLocator(0.1))

    ax.taxis.set_tick_params(tick2On=True, colors=color_map[axis_map["t"]], grid_color=color_map[axis_map["t"]])
    ax.laxis.set_tick_params(tick2On=True, colors=color_map[axis_map["l"]], grid_color=color_map[axis_map["l"]])
    ax.raxis.set_tick_params(tick2On=True, colors=color_map[axis_map["r"]], grid_color=color_map[axis_map["r"]])

    ticks = [0.0, 0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 1]
    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    ax.taxis.set_ticks(ticks, labels=labels)
    ax.laxis.set_ticks(ticks, labels=labels)
    ax.raxis.set_ticks(ticks, labels=labels)

    pc = ax.scatter(r, d, p, c=value, s=500)

    # Add color bar
    cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
    cbar = fig.colorbar(pc, cax=cax)
    pc.set_clim(clim_map[measurement])
    cbar.set_label(measurement)
    if measurement == "rank":
        cbar.ax.invert_xaxis()

    # label
    ax.set_tlabel("R")
    ax.set_llabel("D")
    ax.set_rlabel("P")

    ax.taxis.set_label_position("tick1")
    ax.laxis.set_label_position("tick1")
    ax.raxis.set_label_position("tick1")

    plt.title(output_file_name)

    plt.tight_layout()
    plt.savefig(output_path + output_file_name + "_avg_" + measurement + "_tri.png")
    plt.clf()
    plt.close()


def main():
    """
    # NN
    output_path = OUTPUT_FILE_PATH
    file_name = "r0_d0_p9_T100"
    for measurement in ["cash", "win_rate", "rank", "winning_percentage"]:
        errorbar_plot(f"{file_name}.csv", measurement, output_path)
        tri_plot(f"{file_name}.csv", measurement, output_path)

    """

    for nov in novelty_list:
        # create folder
        # output_path = OUTPUT_FILE_PATH + nov + "/"
        output_path = OUTPUT_FILE_PATH + nov + "/" + model_name + "/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # file_name = f"{model_name}_{nov}"
        # file_name = f"pre_post_r0_d0_p9_T100_card_strength_lr0_01_{nov}"
        for measurement in ["cash", "win_rate", "rank", "winning_percentage"]:
            errorbar_plot(model_name, nov, measurement, output_path)
            # tri_plot(f"{file_name}.csv", measurement, output_path)


if __name__ == "__main__":
    main()
