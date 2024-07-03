"""
This is the rule-based agent of agent_p replication.
For convenience, I am using the methods in gym_open_poker.envs.poker_util and gym_open_poker object directly.
For example, I will decode card 1 into a card object with spade 1.

I recommend that users create their own methods to decode the observation.
"""

from open_world_expt import OpenWorldExpt
from open_world_expt.owe_agent import OweAgent
import numpy as np
import matplotlib.pyplot as plt
import scipy

# for agent calculation
from gym_open_poker.envs.poker_util.action_choices import *
from gym_open_poker.envs.poker_util.card_utility_actions import *
from itertools import combinations
from gym_open_poker.envs.poker_util.agent_helper_function import format_float_precision, get_out_probability, is_out_in_hand
from collections import Counter
from gym_open_poker.envs.poker_util.phase import *
from gym_open_poker.envs.poker_util.card import Card
from agents import AgentDQNReplayFrame
import yaml
import pandas as pd
import os

OUTPUT_FOLDER = "0625_dqn_agent_testing/"
OUTPUT_PATH = f"./agents/results/{OUTPUT_FOLDER}"
# OUTPUT_FILE_NAME = "0306_expt_testing_sum.csv"
TOTAL_TOURNAMENT = 100


if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)


def testing(config_dict, model_name, folder_name=None, NN_count=None, N_count=None, random_seed=20, N_ratio=0.0):

    owe_agent = AgentDQNReplayFrame()

    owe_agent.initialize(learning=False, read_dqn_path=f"./agents/results/{model_name}.keras")

    if NN_count is not None and N_count is not None:
        owe = OpenWorldExpt(owe_agent, NN_count=NN_count, N_count=N_count, N_ratio=N_ratio, random_seed=random_seed)
    else:
        owe = OpenWorldExpt(owe_agent, N_ratio=0.0, random_seed=random_seed)

    owe.execute(config_dict=config_dict, folder_name=folder_name)

    win_list, game_summary = owe.get_summary()
    # print(win_list)
    # print(game_summary[0])
    # print(game_summary[3])
    win_mean, win_std = owe.get_winning_rate()
    """
    print("------")
    print("confusion_matrix")
    owe.get_detection_confusion_matrix(output_format="percentage")
    """

    # owe.get_detection_confusion_matrix(output_format='count')
    # owe.get_summary_figure()
    aggregation_cash_dict = owe.get_model_cash_summary()
    aggregation_rank_dict = owe.get_model_rank_summary()
    aggregation_win_rate_dict = owe.get_model_win_rate_summary()
    aggregation_winning_percentage_dict = owe.get_model_winning_percentage_summary()

    owe.get_player_action_profile_summary()
    owe.get_player_action_percentage_profile_summary()

    pre_post_cash_summary = owe.get_pre_post_model_summary("cash")
    pre_post_rank_summary = owe.get_pre_post_model_summary("rank")
    pre_post_win_rate_summary = owe.get_pre_post_model_summary("win_rate")
    pre_post_winning_percentage_summary = owe.get_pre_post_model_summary("winning_percentage")

    # get ranking average
    average_cash_list = []
    std_cash_list = []
    idx_list = []
    for tournament_idx in range(len(game_summary)):
        tem_cash_list = []
        idx_list.append(tournament_idx)
        for game in range(1, max(game_summary[tournament_idx]["cash"].keys()) + 1):
            tem_cash_list.append(game_summary[tournament_idx]["cash"][game][0])

        average_cash_list.append(np.mean(tem_cash_list))
        std_cash_list.append(np.std(tem_cash_list))

    # statistic, pvalue = scipy.stats.ttest_rel(average_cash_list[:NN_count], average_cash_list[NN_count:])
    # ttest_string = "statistic: " + str(round(statistic, 2)) + ', pvalue = ' + str(round(pvalue, 8))

    """
    plt.errorbar(idx_list, average_cash_list, std_cash_list, ls='none', marker='o')
    plt.axvline(x=30, color='red')
    # plt.gca().invert_yaxis()
    plt.xlabel('Tournament')
    plt.ylabel('Cash')
    # plt.annotate(ttest_string, xy=(0.8, 0.8))
    plt.savefig(folder_name + '_s' + str(random_seed) + '.png')
    plt.close()
    """
    output_dict = dict()
    output_dict["win_mean_list"] = win_mean
    output_dict["win_std_list"] = win_std
    output_dict.update(aggregation_cash_dict)
    output_dict.update(aggregation_rank_dict)
    output_dict.update(aggregation_win_rate_dict)
    output_dict.update(aggregation_winning_percentage_dict)

    pre_post_dict = dict()
    pre_post_dict.update(pre_post_cash_summary)
    pre_post_dict.update(pre_post_rank_summary)
    pre_post_dict.update(pre_post_win_rate_summary)
    pre_post_dict.update(pre_post_winning_percentage_summary)

    return output_dict, pre_post_dict


def expt_summary(model, agent_random_num, agent_dump_num, agent_p_num, output_file_name, novelty_name, random_seed=20):
    with open("./config.yaml", "r") as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    param = f"r{agent_random_num}_d{agent_dump_num}_p{agent_p_num}"
    config_dict["background_agents_raw_list"] = [
        {"agent_type": "agent_p", "number_of_agent": agent_p_num},
        {"agent_type": "agent_dump", "number_of_agent": agent_dump_num},
        {"agent_type": "agent_random", "number_of_agent": agent_random_num},
    ]

    # novelty
    if novelty_name != "NN":
        config_dict["novelty_list"] = [novelty_name]
    folder_name = f"{OUTPUT_FOLDER}/{novelty_name}/{model}/novelty_testing_model_{model}-param_{param}-novelty_{novelty_name}_seed{random_seed}"
    # folder_name = "testing_model_" + model + "-param_" + param
    # win_mean, win_std, detailed_win_rate = testing(config_dict, model, NN_count=50, N_count=50)
    if novelty_name != "NN":
        output_dict, pre_post_dict = testing(
            config_dict, model, folder_name=folder_name, NN_count=NN_count, N_count=N_count, N_ratio=1
        )
    else:
        output_dict, pre_post_dict = testing(
            config_dict, model, folder_name=folder_name, NN_count=TOTAL_TOURNAMENT, N_count=0, N_ratio=0
        )
    output_dict.update({"model": model, "backgound_agent_list": param, "novelty": novelty_name})
    pre_post_dict.update({"model": model, "backgound_agent_list": param, "novelty": novelty_name})
    for item in output_dict:
        output_dict[item] = [output_dict[item]]

    for item in pre_post_dict:
        pre_post_dict[item] = [pre_post_dict[item]]

    output_df = pd.DataFrame(output_dict)
    if os.path.exists(OUTPUT_PATH + output_file_name):
        output_df.to_csv(OUTPUT_PATH + output_file_name, mode="a", header=False)
    else:
        output_df.to_csv(OUTPUT_PATH + output_file_name)

    pre_post_dict_df = pd.DataFrame(pre_post_dict)
    if os.path.exists(OUTPUT_PATH + "pre_post_" + output_file_name):
        pre_post_dict_df.to_csv(OUTPUT_PATH + "pre_post_" + output_file_name, mode="a", header=False)
    else:
        pre_post_dict_df.to_csv(OUTPUT_PATH + "pre_post_" + output_file_name)


def main():
    """
    for model in ["r9_d0_p0_T100", "r0_d9_p0_T100", "r0_d0_p9_T100"]:
        agent_p_num = 0
        for agent_random_num in range(9, -1, -1):
            agent_dump_num = 9 - agent_random_num
            output_file_name = f"{model}.csv"
            expt_summary(model, agent_random_num, agent_dump_num, agent_p_num, output_file_name)

    # for model in ["r9_d0_p0_T100", "r0_d9_p0_T100", "r0_d0_p9_T100"]:
    for model in ["r0_d9_p0_T100"]:
        agent_random_num = 0
        for agent_dump_num in range(9, -1, -1):
            agent_p_num = 9 - agent_dump_num
            output_file_name = f"{model}.csv"
            expt_summary(model, agent_random_num, agent_dump_num, agent_p_num, output_file_name)

    for model in ["r0_d0_p9_T100"]:
        for agent_random_num in range(9, -1, -1):
            for agent_dump_num in range(9 - agent_random_num, -1, -1):
                agent_p_num = 9 - agent_random_num - agent_dump_num
                output_file_name = f"{model}.csv"
                expt_summary(model, agent_random_num, agent_dump_num, agent_p_num, output_file_name)

    for model in ["r0_d0_p9_T100_card"]:
        for agent_random_num, agent_dump_num, agent_p_num in [(9, 0, 0), (0, 9, 0), (0, 0, 9), (3, 3, 3)]:
            output_file_name = f"{model}.csv"
            expt_summary(model, agent_random_num, agent_dump_num, agent_p_num, output_file_name)
    """
    novelty_name = "NN"
    seed = 20
    for model in ["0517_r0_d0_p9_T300_replay_frame_lr0.0001_delta0.9"]:
        # for agent_random_num, agent_dump_num, agent_p_num in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        for agent_random_num, agent_dump_num, agent_p_num in [
            (7, 1, 1),
            (1, 7, 1),
            (1, 1, 7),
            (5, 2, 2),
            (2, 5, 2),
            (2, 2, 5),
            (3, 3, 3),
        ]:
            """
            for agent_random_num, agent_dump_num, agent_p_num in [
                (9, 0, 0),
                (0, 9, 0),
                (0, 0, 9),
                (3, 3, 3),
                (7, 1, 1),
                (1, 7, 1),
                (1, 1, 7),
                (5, 2, 2),
                (2, 5, 2),
                (2, 2, 5),
            ]:
            """
            output_file_name = f"{model}_{novelty_name}_seed{seed}.csv"
            expt_summary(
                model, agent_random_num, agent_dump_num, agent_p_num, output_file_name, novelty_name, random_seed=seed
            )


if __name__ == "__main__":
    main()
