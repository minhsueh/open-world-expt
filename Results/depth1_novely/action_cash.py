import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.lines as mlines


# The performance of post-novelty is extracted from:
# ./depth1_novely/agents/results/0520_dqn_agent_testing/action.GameFoldRestrict_seed20/0517_r0_d0_p9_T300_replay_frame_lr0.0001_delta0.9/pre_post_0517_r0_d0_p9_T300_replay_frame_lr0.0001_delta0.9_action.GameFoldRestrict_seed20.csv

x = ["Pre-", "Post-"]
DQN_R = [592.0, 263.37]
DQN_R_ste = [100.11, 65.26]
DQN_P = [551.75, 121.68]
DQN_P_ste = [73.91, 33.89]
DQN_D = [170.75, 237.65]
DQN_D_ste = [57.52, 45.82]
R = [143.86, 190.94]
R_ste = [14.32, 83.39]
D = [204.17, 194.62]
D_ste = [8.21, 6.56]
P = [149.71, 211.18]
P_ste = [10.55, 4.84]

color_dict = {"player_1": "red", "agent_random": "green", "agent_dump": "darkviolet", "agent_p": "blue"}
R_dict = {"player_1": [DQN_R, DQN_R_ste], "agent_random": [R, R_ste]}
D_dict = {"player_1": [DQN_D, DQN_D_ste], "agent_dump": [D, D_ste]}
P_dict = {"player_1": [DQN_P, DQN_P_ste], "agent_p": [P, P_ste]}
data_dict = {"agent_random": R_dict, "agent_dump": D_dict, "agent_p": P_dict}
label_dict = {"agent_random": "AgentR", "agent_dump": "AgentD", "agent_p": "AgentP"}


for ba in ["agent_random", "agent_dump", "agent_p"]:
    ba_dict = data_dict[ba]
    DQN, DQN_ste = ba_dict["player_1"]
    BA, BA_ste = ba_dict[ba]

    plt.errorbar(x, DQN, DQN_ste, c="red", capsize=5, fmt="x", label="AgentDQN", ls="-")
    plt.errorbar(x, BA, BA_ste, color=color_dict[ba], capsize=5, fmt=".", label=label_dict[ba], ls="-")

    plt.legend(loc=1)
    plt.grid()
    # plt.tight_layout()
    plt.ylabel("Cash")
    plt.ylim([0, 1000])
    plt.savefig(
        ba + "_action.png",
    )
    plt.cla()
