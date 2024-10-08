import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.lines as mlines


OUTPUT_FOLDER_PATTERN = "0529_dqn_agent_testing_{exp_idx}/"
model_name = "0517_r0_d0_p9_T300_replay_frame_lr0.0001_delta0.9"

# OUTPUT_FILE_PATH = f"./agents/results/{OUTPUT_FOLDER}"
# SINGLE_OUTPUT_FILE_PATH = f"./agents/results/{SINGLE_OUTPUT_FOLDER}"
output_df = pd.DataFrame({"AgentDQN": [], "AgentDQN_std": [], "BA": [], "BA_std": [], "depth": [], "novelty": []})
DQN_pre_cash_dict = {"r7_d0_p0": [759.35, 51.35], "r0_d7_p0": [185.13, 31.56], "r0_d0_p7": [586.05, 35.66]}
BA_pre_cash_dict = {"r7_d0_p0": [119.88, 7.34], "r0_d7_p0": [201.98, 4.50], "r0_d0_p7": [144.88, 4.81]}
agent_dict = {"r7_d0_p0": "agent_random", "r0_d7_p0": "agent_dump", "r0_d0_p7": "agent_p"}
model_dict = {"r7_d0_p0": "AgentR", "r0_d7_p0": "AgentD", "r0_d0_p7": "AgentP"}

for background_agent in ["r7_d0_p0", "r0_d7_p0", "r0_d0_p7"]:

    AgentDQN_All = []
    AgentDQN_ste_All = []
    BA_All = []
    BA_ste_All = []
    depth_All = []
    novelty_All = []
    expt_All = []

    for exp_idx in range(1, 11):
        AgentDQN = [DQN_pre_cash_dict[background_agent][0]]
        AgentDQN_ste = [DQN_pre_cash_dict[background_agent][1]]
        BA = [BA_pre_cash_dict[background_agent][0]]
        BA_ste = [BA_pre_cash_dict[background_agent][1]]
        depth = [0]
        novelty = [""]
        expt = [exp_idx]

        OUTPUT_FOLDER = OUTPUT_FOLDER_PATTERN.format(exp_idx=exp_idx)
        OUTPUT_FILE_PATH = f"./agents/results/{OUTPUT_FOLDER}"

        folder_list = [file for file in os.listdir(OUTPUT_FILE_PATH)]
        folder_list.sort(key=lambda x: len(x))
        for folder in folder_list:
            csv_file_folder = OUTPUT_FILE_PATH + folder + "/" + model_name + "/"
            d = max(len(folder.split("+")), 1)
            for file in os.listdir(csv_file_folder):
                if file.startswith("pre_post"):
                    break

            df = pd.read_csv(csv_file_folder + file)
            tem_df = df[df["backgound_agent_list"] == background_agent]
            AgentDQN.append(tem_df["post_player_1_cash"].values[0])
            AgentDQN_ste.append(tem_df["post_player_1_cash_ste"].values[0])
            BA.append(tem_df["post_" + agent_dict[background_agent] + "_cash"].values[0])
            BA_ste.append(tem_df["post_" + agent_dict[background_agent] + "_cash_ste"].values[0])
            depth.append(d)
            novelty.append(folder)
            expt.append(exp_idx)

        AgentDQN_All += AgentDQN
        AgentDQN_ste_All += AgentDQN_ste
        BA_All += BA
        BA_ste_All += BA_ste
        depth_All += depth
        novelty_All += novelty
        expt_All += expt

    output_dict = {
        "AgentDQN": AgentDQN_All,
        "AgentDQN_ste": AgentDQN_ste_All,
        model_dict[background_agent]: BA_All,
        model_dict[background_agent] + "_ste": BA_ste_All,
        "depth": depth_All,
        "novelty": novelty_All,
        "expt": expt_All,
    }  # novelty, novelty_idx, depth, DQN_R, DQN_R_ste, R, R_ste
    output_df = pd.DataFrame(output_dict)
    output_df.to_csv("./agents/results/" + "depth_DQN_against" + model_dict[background_agent] + ".csv")

    # plot
    for exp_idx in range(1, 11):
        tem_df = output_df[output_df["expt"] == exp_idx]
        plt.errorbar(
            tem_df["depth"],
            tem_df["AgentDQN"],
            tem_df["AgentDQN_ste"],
            capsize=5,
            fmt="x",
            ls="dotted",
            label=exp_idx,
            alpha=0.5,
        )
    plt.grid()
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.axhline(y=DQN_pre_cash_dict[background_agent][0], color="black", linestyle="--", alpha=0.5)
    plt.xlabel("Depth")
    plt.ylabel("Cash")
    plt.tight_layout()
    plt.ylim([0, 1200])
    plt.savefig("./agents/results/" + "depth_DQN_against" + model_dict[background_agent] + ".png")
    plt.cla()
