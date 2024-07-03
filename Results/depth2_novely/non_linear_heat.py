import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.lines as mlines
import seaborn as sns
from matplotlib.tri import Triangulation

SINGLE_OUTPUT_FOLDER = "0520_dqn_agent_testing/"
OUTPUT_FOLDER = "0524_dqn_agent_testing/"
model_name = "0517_r0_d0_p9_T300_replay_frame_lr0.0001_delta0.9"
"""
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
"""
novelty_list = [
    "card.CardDistHigh_seed20",
    "card.CardDistLow_seed20",
    "card.CardDistSuit_seed20",
    "card.CardDistOdd_seed20",
    "card.CardDistColor_seed20",
    "conclude_game.Incentive_seed20",
    "conclude_game.Tipping_seed20",
    "conclude_game.SeatChanging_seed20",
    "conclude_game.LuckySeven_seed20",
    "conclude_game.HiddenCard_seed20",
    "game_element.AllOdd_seed20",
    "game_element.BigBlindChange_seed20",
    "game_element.BigBetChange_seed20",
    "game_element.BuyIn_seed20",
    "game_element.TournamentLength_seed20",
    "agent.AgentExchange_seed20",
    "agent.AddAgentR_seed20",
    "agent.AddAgentConservative_seed20",
    "agent.AddAgentAggressive_seed20",
    "agent.AddAgentStirring_seed20",
    "action.GameFoldRestrict_seed20",
    "action.NoFreeLunch_seed20",
    "action.ActionHierarchy_seed20",
    "action.WealthTax_seed20",
    "action.RoundActionReStrict_seed20",
]

novelty_dict = {
    "CardDistHigh": "c1",
    "CardDistLow": "c2",
    "CardDistSuit": "c3",
    "CardDistOdd": "c4",
    "CardDistColor": "c5",
    "Incentive": "cg1",
    "Tipping": "cg2",
    "SeatChanging": "cg3",
    "LuckySeven": "cg4",
    "HiddenCard": "cg5",
    "AllOdd": "g1",
    "BigBlindChange": "g2",
    "BigBetChange": "g3",
    "BuyIn": "g4",
    "TournamentLength": "g5",
    "AgentExchange": "ag1",
    "AddAgentR": "ag2",
    "AddAgentConservative": "ag3",
    "AddAgentAggressive": "ag4",
    "AddAgentStirring": "ag5",
    "GameFoldRestrict": "ac1",
    "NoFreeLunch": "ac2",
    "ActionHierarchy": "ac3",
    "WealthTax": "ac4",
    "RoundActionReStrict": "ac5",
}

# novelty_id_list = [""] + [nov.split("_seed20")[0].split(".")[1] for nov in novelty_list]
novelty_id_list = [""] + [novelty_dict[nov] for nov in novelty_dict]

# base_novelty = "conclude_game.HiddenCard"
# base_novelty = "card.CardDistSuit"

OUTPUT_FILE_PATH = f"./agents/results/{OUTPUT_FOLDER}"
SINGLE_OUTPUT_FILE_PATH = f"./agents/results/{SINGLE_OUTPUT_FOLDER}"

# background_agent = "agent_p"
# model_list = ["player_1", "agent_random", "agent_dump", "agent_p"]

# novelty_list = ["NN"]

csv_file_pattern = OUTPUT_FILE_PATH + "{novelty}" + "/" + model_name + "/" + "pre_post_" + model_name + "_{novelty}.csv"
single_csv_file_pattern = (
    SINGLE_OUTPUT_FILE_PATH + "{novelty}" + "/" + model_name + "/" + "pre_post_" + model_name + "_{novelty}.csv"
)

measurement_list = ["cash", "win_rate", "rank", "winning_percentage"]
measurement = "cash"

name_dict = {"r7_d0_p0": "AgentR", "r0_d7_p0": "AgentD", "r0_d0_p7": "AgentP"}
name_to_column_dict = {"r7_d0_p0": "agent_random", "r0_d7_p0": "agent_dump", "r0_d0_p7": "agent_p"}
column_to_name_dict = {"agent_random": "r7_d0_p0", "agent_dump": "r0_d7_p0", "agent_p": "r0_d0_p7"}
legend_name_dict = {"agent_random": "AgentR", "agent_dump": "AgentD", "agent_p": "AgentP"}


model_list = []
color_list = ["red", "green", "blue", "darkviolet"]
color_dict = {"player_1": "red", "agent_random": "green", "agent_dump": "darkviolet", "agent_p": "blue"}


for background_agent in ["agent_random", "agent_dump", "agent_p"]:

    player_1_pre_mean = []
    player_1_pre_std = []

    player_1_post_nov = []
    player_1_post_mean = []
    player_1_post_ste = []

    background_pre_mean = []
    background_pre_std = []

    background_post_mean = []
    background_post_ste = []

    output_df = pd.DataFrame(columns=novelty_id_list, index=novelty_id_list)

    # aggregate cash data for background_agent
    for idx in range(len(novelty_list)):
        for jdx in range(idx + 1, len(novelty_list)):
            base_novelty_raw = novelty_list[idx]
            novelty_raw = novelty_list[jdx]
            #
            nc1 = base_novelty_raw.split(".")[0]
            nc2 = novelty_raw.split(".")[0]
            if nc1 == nc2:
                continue

            #
            # novelty1 = base_novelty + "+" + novelty_raw
            novelty1 = base_novelty_raw.split("_seed20")[0] + "+" + novelty_raw
            novelty2 = novelty_raw.split("_seed20")[0] + "+" + base_novelty_raw
            if os.path.isfile(csv_file_pattern.format(novelty=novelty1)):
                novelty = novelty1
            elif os.path.isfile(csv_file_pattern.format(novelty=novelty2)):
                novelty = novelty2
            else:
                print(novelty1, column_to_name_dict[background_agent])
                continue
                # raise
            csv_file_name = csv_file_pattern.format(novelty=novelty)
            df = pd.read_csv(csv_file_name)
            tem_df = df[df["backgound_agent_list"] == column_to_name_dict[background_agent]]
            if tem_df.empty or tem_df.shape[0] > 1:
                print(novelty, column_to_name_dict[background_agent])
                raise

            nov1 = base_novelty_raw.split("_seed20")[0].split(".")[1]
            nov2 = novelty_raw.split("_seed20")[0].split(".")[1]

            #
            nov1 = novelty_dict[nov1]
            nov2 = novelty_dict[nov2]

            output_df.loc[nov1, nov2] = tem_df["post_player_1_cash"].values[0]
            output_df.loc[nov2, nov1] = tem_df["post_" + background_agent + "_cash"].values[0]

    # diagomal, single novelty
    for idx in range(len(novelty_list)):
        base_novelty_raw = novelty_list[idx]
        nov1 = base_novelty_raw.split("_seed20")[0].split(".")[1]

        nov1 = novelty_dict[nov1]

        # get single novlety
        single_csv_file_name = single_csv_file_pattern.format(novelty=base_novelty_raw)
        single_df = pd.read_csv(single_csv_file_name)
        tem_single_df = single_df[single_df["backgound_agent_list"] == column_to_name_dict[background_agent]]
        # player_1
        output_df.loc["", nov1] = tem_single_df["post_player_1_cash"].values[0]
        # background_agent
        output_df.loc[nov1, ""] = tem_single_df["post_" + background_agent + "_cash"].values[0]

    output_df = output_df.astype("float")

    output_df.to_csv(
        OUTPUT_FILE_PATH + "heat" + measurement + "_" + background_agent + ".csv",
    )
    lm = sns.heatmap(output_df, xticklabels=True, yticklabels=True, linecolor="k", linewidths=0.5, vmin=0, vmax=2000)
    ax = lm.axes
    ax.tick_params(labelbottom=False, labeltop=True)
    plt.xticks(rotation=90)
    ax.axline([ax.get_xlim()[0], ax.get_ylim()[1]], [ax.get_xlim()[1], ax.get_ylim()[0]], color="grey", lw=0.5)
    nc_dict = {1: "Card", 6: "Conclude\nGame", 11: "Game\nElement", 16: "Agent", 21: "Action"}
    for idx in range(1, 26, 5):
        ax.axhline(y=idx, color="lightgrey", linestyle="-", lw=0.5)
        ax.axvline(x=idx, color="lightgrey", linestyle="-", lw=0.5)
        plt.text(
            idx + 2.5,
            idx + 2.5,
            nc_dict[idx],
            fontdict=dict(fontsize=8, fontweight="bold"),
            bbox=dict(facecolor="lightgrey", edgecolor="black"),
            ha="center",
            va="center",
        )
        # ax.annotate(nc_dict[idx], xy=(idx, idx + 3), fill="green")
    # plt.grid()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_FILE_PATH + "heat" + measurement + "_" + background_agent + ".png",
    )
    plt.clf()

    output_df.to_csv(
        OUTPUT_FILE_PATH + "heat" + measurement + "_" + background_agent + ".csv",
    )
