import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.lines as mlines


OUTPUT_FOLDER = "0520_dqn_agent_testing/"
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

OUTPUT_FILE_PATH = f"./agents/results/{OUTPUT_FOLDER}"

# background_agent = "agent_p"
# model_list = ["player_1", "agent_random", "agent_dump", "agent_p"]

# novelty_list = ["NN"]

csv_file_pattern = OUTPUT_FILE_PATH + "{novelty}" + "/" + model_name + "/" + "pre_post_" + model_name + "_{novelty}.csv"

measurement_list = ["cash", "win_rate", "rank", "winning_percentage"]
measurement = "cash"

name_dict = {"r7_d0_p0": "AgentR", "r0_d7_p0": "AgentD", "r0_d0_p7": "AgentP"}
name_to_column_dict = {"r7_d0_p0": "agent_random", "r0_d7_p0": "agent_dump", "r0_d0_p7": "agent_p"}
column_to_name_dict = {"agent_random": "r7_d0_p0", "agent_dump": "r0_d7_p0", "agent_p": "r0_d0_p7"}
legend_name_dict = {"agent_random": "AgentR", "agent_dump": "AgentD", "agent_p": "AgentP"}

pre_pooled_mean_dict = {"player_1": "red", "agent_random": "green", "agent_dump": "darkviolet", "agent_p": "blue"}

model_list = []
color_list = ["red", "green", "blue", "darkviolet"]
color_dict = {"player_1": "red", "agent_random": "green", "agent_dump": "darkviolet", "agent_p": "blue"}

DQN_pre_cash_dict = {"agent_random": [759.35, 51.35], "agent_dump": [185.13, 31.56], "agent_p": [586.05, 35.66]}
BA_pre_cash_dict = {"agent_random": [119.88, 7.34], "agent_dump": [201.98, 4.50], "agent_p": [144.88, 4.81]}

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

    # aggregate cash data for background_agent
    for novelty in novelty_list:
        csv_file_name = csv_file_pattern.format(novelty=novelty)
        df = pd.read_csv(csv_file_name)
        tem_df = df[df["backgound_agent_list"] == column_to_name_dict[background_agent]]

        # player_1 pre-novelty
        player_1_pre_mean.append(tem_df["pre_player_1_cash"].values[0])
        player_1_pre_std.append(tem_df["pre_player_1_cash_std"].values[0])

        # get player_1 post novelty
        novelty_id = novelty.split(".")[1].split("_")[0]
        # player_1_post_nov.append(novelty_id)
        player_1_post_nov.append(novelty_dict[novelty_id])
        player_1_post_mean.append(tem_df["post_player_1_cash"].values[0])
        player_1_post_ste.append(tem_df["post_player_1_cash_ste"].values[0])

        # background pre-novelty
        background_pre_mean.append(tem_df["pre_" + background_agent + "_cash"].values[0])
        background_pre_std.append(tem_df["pre_" + background_agent + "_cash_std"].values[0])

        # get background post novelty
        background_post_mean.append(tem_df["post_" + background_agent + "_cash"].values[0])
        background_post_ste.append(tem_df["post_" + background_agent + "_cash_ste"].values[0])

    player_1_pre_pooled_mean = np.mean(player_1_pre_mean)
    np_player_1_pre_std = np.array(player_1_pre_std)
    player_1_pre_pooled_ste = (sum(np_player_1_pre_std * np_player_1_pre_std)) ** 0.5 / len(np_player_1_pre_std)

    background_pre_pooled_mean = np.mean(background_pre_mean)
    np_background_pre_std = np.array(background_pre_std)
    background_pre_pooled_ste = (sum(np_background_pre_std * np_background_pre_std)) ** 0.5 / len(np_background_pre_std)

    zipped = zip(player_1_post_nov, player_1_post_mean, player_1_post_ste, background_post_mean, background_post_ste)
    zipped_sort = sorted(zipped, reverse=True, key=lambda x: x[1])
    (
        player_1_post_nov_sort,
        player_1_post_mean_sort,
        player_1_post_ste_sort,
        background_post_mean_sort,
        background_post_ste_sort,
    ) = zip(*zipped_sort)

    plt.errorbar(
        [""],
        # [player_1_pre_pooled_mean],
        # [player_1_pre_pooled_ste],
        DQN_pre_cash_dict[background_agent][0],
        DQN_pre_cash_dict[background_agent][1],
        c="red",
        capsize=5,
        fmt="x",
    )

    plt.errorbar(
        [""],
        # [background_pre_pooled_mean],
        # [background_pre_pooled_ste],
        BA_pre_cash_dict[background_agent][0],
        BA_pre_cash_dict[background_agent][1],
        c=color_dict[background_agent],
        capsize=5,
        fmt="x",
    )

    plt.errorbar(
        player_1_post_nov_sort,
        player_1_post_mean_sort,
        player_1_post_ste_sort,
        c="red",
        capsize=5,
        fmt=".",
        # label=name_dict[row["backgound_agent_list"]],
    )

    plt.errorbar(
        player_1_post_nov_sort,
        background_post_mean_sort,
        background_post_ste_sort,
        c=color_dict[background_agent],
        capsize=5,
        fmt=".",
        # label=name_dict[row["backgound_agent_list"]],
    )

    # label
    pre_circle = mlines.Line2D([], [], color="black", marker="o", linestyle="None", markersize=6, label="Pre")
    post_cross = mlines.Line2D([], [], color="black", marker="x", linestyle="None", markersize=6, label="Post")
    pre_square = mlines.Line2D([], [], color="black", marker="s", alpha=0.3, linestyle="None", markersize=6, label="Pre")
    post_square = mlines.Line2D([], [], color="white", marker="s", alpha=0.3, linestyle="None", markersize=6, label="Post")
    red_square = mlines.Line2D([], [], color="red", marker="s", linestyle="None", markersize=6, label="AgentDQN")
    background_square = mlines.Line2D(
        [],
        [],
        color=color_dict[background_agent],
        marker="s",
        linestyle="None",
        markersize=6,
        label=legend_name_dict[background_agent],
    )

    plt.legend(
        # handles=[pre_circle, post_cross, pre_square, red_square, background_square],
        handles=[pre_square, red_square, background_square],
        # loc="center left",
        loc=1,
        # bbox_to_anchor=(1, 0.5),
    )
    plt.xticks(rotation=90)
    # ax = plt.gca()
    # xlbl = ax.xaxis.get_label()
    # print(xlbl)
    plt.axhline(y=player_1_pre_pooled_mean, color="black", linestyle="--", alpha=0.5)
    plt.ylabel("Cash")
    plt.ylim([0, 1000])
    plt.axvspan(-0.5, 0.5, facecolor="black", alpha=0.1)
    plt.grid()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_FILE_PATH + measurement + "_" + background_agent + ".png",
    )
    plt.clf()

    df = pd.DataFrame(
        {
            "player_1_post_nov_sort": player_1_post_nov_sort,
            "player_1_post_mean_sort": player_1_post_mean_sort,
            "player_1_post_ste_sort": player_1_post_ste_sort,
            "background_post_mean_sort": background_post_mean_sort,
            "background_post_ste_sort": background_post_ste_sort,
        }
    )
    df.to_csv(
        OUTPUT_FILE_PATH + measurement + "_" + background_agent + ".csv",
    )
