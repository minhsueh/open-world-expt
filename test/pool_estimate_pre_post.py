import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# OUTPUT_FOLDER = "0627_dqn_agent_testing/"
OUTPUT_FOLDER = "20260509_dqn_agent_testing_3/"
model_name = "0517_r0_d0_p9_T300_replay_frame_lr0.0001_delta0.9"
novelty_list = ["action.GameFoldRestrict"]
OUTPUT_FILE_PATH = f"./agents/results/{OUTPUT_FOLDER}"

# novelty_list = ["NN"]

csv_file_pattern = (
    OUTPUT_FILE_PATH
    + "{novelty}"
    + "/"
    + model_name
    + "/"
    + "pre_post_"
    + model_name
    + "_{novelty}.csv"
)

measurement_list = ["cash", "win_rate", "rank", "winning_percentage"]
model_list = ["player_1", "agent_random", "agent_dump", "agent_p"]

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
            pre_pooled_std = (
                sum(pre_std_list * pre_std_list) / len(pre_std_list)
            ) ** 0.5
            pre_pooled_ste = (sum(pre_std_list * pre_std_list)) ** 0.5 / len(
                pre_std_list
            )
            post_pooled_mean = np.mean(post_mean_list)
            post_pooled_std = (
                sum(post_std_list * post_std_list) / len(post_std_list)
            ) ** 0.5
            post_pooled_ste = (sum(post_std_list * post_std_list)) ** 0.5 / len(
                post_std_list
            )

            output_dict["pre_" + measurement + "_pooled_mean"].append(pre_pooled_mean)
            output_dict["pre_" + measurement + "_pooled_std"].append(pre_pooled_std)
            output_dict["pre_" + measurement + "_pooled_ste"].append(pre_pooled_ste)
            output_dict["pre_" + measurement + "_pooled_dof"].append(len(pre_std_list))
            output_dict["post_" + measurement + "_pooled_mean"].append(post_pooled_mean)
            output_dict["post_" + measurement + "_pooled_std"].append(post_pooled_std)
            output_dict["post_" + measurement + "_pooled_ste"].append(post_pooled_ste)
            output_dict["post_" + measurement + "_pooled_dof"].append(
                len(post_std_list)
            )

output_df = pd.DataFrame(output_dict)


# plot

for novelty in novelty_list:
    for measurement in measurement_list:
        if not os.path.exists(OUTPUT_FILE_PATH + novelty + "/" + model_name):
            os.makedirs(OUTPUT_FILE_PATH + novelty + "/" + model_name)

        tem_df = output_df[output_df["novelty"] == novelty]
        pre_x = [0.3, 2.3, 4.3, 6.3]
        post_x = [0.7, 2.7, 4.7, 6.7]
        x_ticks = [0.5, 2.5, 4.5, 6.5]

        plt.errorbar(
            pre_x,
            tem_df["pre_" + measurement + "_pooled_mean"],
            tem_df["pre_" + measurement + "_pooled_ste"],
            capsize=5,
            fmt="o",
            label="Pre",
        )
        plt.errorbar(
            post_x,
            tem_df["post_" + measurement + "_pooled_mean"],
            tem_df["post_" + measurement + "_pooled_ste"],
            capsize=5,
            fmt="x",
            label="Post",
        )
        plt.xticks(x_ticks, model_list)
        plt.xlabel("Model")
        plt.ylabel(measurement.capitalize())
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
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
            + ".png",
        )
        plt.clf()

    output_df.to_csv(
        OUTPUT_FILE_PATH
        + novelty
        + "/"
        + model_name
        + "/"
        + model_name
        + "_pre_post_pooled_estimate.csv"
    )
    # OUTPUT_FILE_PATH
