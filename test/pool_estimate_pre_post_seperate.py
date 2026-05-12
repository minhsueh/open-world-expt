import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.lines as mlines


# OUTPUT_FOLDER = "0517_dqn_agent_testing/"
OUTPUT_FOLDER = "20260509_dqn_agent_testing_3/"
model_name = "0517_r0_d0_p9_T300_replay_frame_lr0.0001_delta0.9"
novelty_list = [
    "action.GameFoldRestrict_seed20",
    "agent.AgentExchange_seed20",
    "card.CardDistColor_seed20",
    "card.CardDistHigh_seed20",
    "card.CardDistLow_seed20",
    "card.CardDistOdd_seed20",
    "card.CardDistSuit_seed20",
    "conclude_game.Incentive_seed20",
    "conclude_game.LuckySeven_seed20",
    "conclude_game.SeatChanging_seed20",
    "conclude_game.Tipping_seed20",
    "game_element.AllOdd_seed20",
    "game_element.BigBetChange_seed20",
    "game_element.BigBlindChange_seed20",
    "game_element.BuyIn_seed20",
    "game_element.TournamentLength_seed20",
]

novelty_list = ["action.GameFoldRestrict_seed20"]


OUTPUT_FILE_PATH = f"./agents/results/{OUTPUT_FOLDER}"

# novelty_list = ["NN"]

# csv_file_pattern = OUTPUT_FILE_PATH + "{novelty}" + "/" + model_name + "/" + "pre_post_" + model_name + "_{novelty}.csv"
csv_file_pattern = OUTPUT_FILE_PATH + "/" + "pre_post_" + model_name + "_{novelty}.csv"


measurement_list = ["cash", "win_rate", "rank", "winning_percentage"]


"""
output_dict = dict()
output_dict["novelty"] = novelty_list
for measurement in measurement_list:
    output_dict["novelty"] = []
    output_dict["model"] = []
    output_dict["pre_" + measurement + "_pooled_mean"] = []
    output_dict["pre_" + measurement + "_pooled_std"] = []
    output_dict["pre_" + measurement + "_pooled_ste"] = []
    output_dict["pre_" + measurement + "_pooled_dof"] = []
    output_dict["post_" + measurement + "_pooled_mean"] = []
    output_dict["post_" + measurement + "_pooled_std"] = []
    output_dict["post_" + measurement + "_pooled_ste"] = []
    output_dict["post_" + measurement + "_pooled_dof"] = []

for novelty in novelty_list:
    csv_file_name = csv_file_pattern.format(novelty=novelty)
    df = pd.read_csv(csv_file_name)
    print(csv_file_name)

    for model in model_list:
        output_dict["novelty"].append(novelty)
        output_dict["model"].append(model)
        for measurement in measurement_list:
            pre_mean_list = df["pre_" + model + "_" + measurement].dropna()
            pre_std_list = df["pre_" + model + "_" + measurement + "_std"].dropna()
            post_mean_list = df["post_" + model + "_" + measurement].dropna()
            post_std_list = df["post_" + model + "_" + measurement + "_std"].dropna()

            pre_pooled_mean = np.mean(pre_mean_list)
            pre_pooled_std = (sum(pre_std_list * pre_std_list) / len(pre_std_list)) ** 0.5
            pre_pooled_ste = (sum(pre_std_list * pre_std_list)) ** 0.5 / len(pre_std_list)
            post_pooled_mean = np.mean(post_mean_list)
            post_pooled_std = (sum(post_std_list * post_std_list) / len(post_std_list)) ** 0.5
            post_pooled_ste = (sum(post_std_list * post_std_list)) ** 0.5 / len(post_std_list)

            output_dict["pre_" + measurement + "_pooled_mean"].append(pre_pooled_mean)
            output_dict["pre_" + measurement + "_pooled_std"].append(pre_pooled_std)
            output_dict["pre_" + measurement + "_pooled_ste"].append(pre_pooled_ste)
            output_dict["pre_" + measurement + "_pooled_dof"].append(len(pre_std_list))
            output_dict["post_" + measurement + "_pooled_mean"].append(post_pooled_mean)
            output_dict["post_" + measurement + "_pooled_std"].append(post_pooled_std)
            output_dict["post_" + measurement + "_pooled_ste"].append(post_pooled_ste)
            output_dict["post_" + measurement + "_pooled_dof"].append(len(post_std_list))

output_df = pd.DataFrame(output_dict)
"""


name_dict = {"r9_d0_p0": "AgentR", "r0_d9_p0": "AgentD", "r0_d0_p9": "AgentP"}
name_to_column_dict = {
    "r9_d0_p0": "agent_random",
    "r0_d9_p0": "agent_dump",
    "r0_d0_p9": "agent_p",
}
# model_list = ["player_1", "agent_random", "agent_dump", "agent_p"]
model_list = []
color_list = ["red", "green", "blue", "darkviolet"]

# plot

for novelty in novelty_list:
    csv_file_name = csv_file_pattern.format(novelty=novelty)
    df = pd.read_csv(csv_file_name)

    if not os.path.exists(OUTPUT_FILE_PATH + novelty + "/" + model_name):
        os.makedirs(OUTPUT_FILE_PATH + novelty + "/" + model_name)

    for measurement in measurement_list:
        pre_x = [0.2, 0.6, 2.2, 2.6, 4.2, 4.6]
        post_x = [0.4, 0.8, 2.4, 2.8, 4.4, 4.8]
        x_ticks = [0.5, 2.5, 4.5]

        for index, row in df.iterrows():
            if name_dict[row["backgound_agent_list"]] not in model_list:
                model_list.append(name_dict[row["backgound_agent_list"]])

            for offset_idx in range(2):
                if offset_idx == 0:
                    agent_name = "player_1"
                    color = color_list[0]
                else:
                    agent_name = name_to_column_dict[row["backgound_agent_list"]]
                    color = color_list[index + 1]

                plt.errorbar(
                    pre_x[index * 2 + offset_idx],
                    row["pre_" + agent_name + "_" + measurement],
                    row["pre_" + agent_name + "_" + measurement + "_ste"],
                    c=color,
                    capsize=5,
                    fmt="x",
                    # label=name_dict[row["backgound_agent_list"]],
                )
                plt.errorbar(
                    post_x[index * 2 + offset_idx],
                    row["post_" + agent_name + "_" + measurement],
                    row["post_" + agent_name + "_" + measurement + "_ste"],
                    c=color,
                    capsize=5,
                    fmt="x",
                    # label=name_dict[row["backgound_agent_list"]],
                )

        plt.xticks(x_ticks, model_list)
        plt.axvline(x=1.5, color="black")
        plt.axvline(x=3.5, color="black")

        # plt.xlabel("Model")
        plt.ylabel(measurement.capitalize())

        # label
        pre_circle = mlines.Line2D(
            [],
            [],
            color="black",
            marker="o",
            linestyle="None",
            markersize=6,
            label="Pre",
        )
        post_cross = mlines.Line2D(
            [],
            [],
            color="black",
            marker="x",
            linestyle="None",
            markersize=6,
            label="Post",
        )
        red_square = mlines.Line2D(
            [],
            [],
            color="red",
            marker="s",
            linestyle="None",
            markersize=6,
            label="AgentDQN",
        )
        green_square = mlines.Line2D(
            [],
            [],
            color="green",
            marker="s",
            linestyle="None",
            markersize=6,
            label="AgentR",
        )
        blue_square = mlines.Line2D(
            [],
            [],
            color="blue",
            marker="s",
            linestyle="None",
            markersize=6,
            label="AgentD",
        )
        purple_square = mlines.Line2D(
            [],
            [],
            color="darkviolet",
            marker="s",
            linestyle="None",
            markersize=6,
            label="AgentP",
        )

        plt.legend(
            handles=[
                pre_circle,
                post_cross,
                red_square,
                green_square,
                blue_square,
                purple_square,
            ],
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )
        plt.grid()
        plt.tight_layout()
        plt.savefig(
            OUTPUT_FILE_PATH
            + novelty
            + "/"
            + model_name
            + "/"
            + model_name
            + "_pooled_"
            + measurement
            + "_seperate.png",
        )
        plt.clf()

    # output_df.to_csv(OUTPUT_FILE_PATH + novelty + "/" + model_name + "/" + model_name + "_pre_post_pooled_estimate.csv")
    # OUTPUT_FILE_PATH
