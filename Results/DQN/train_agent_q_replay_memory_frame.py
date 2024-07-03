from open_world_expt import OpenWorldExpt
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from itertools import combinations
from collections import Counter
from agents import AgentDQNReplayFrame
import yaml
import time
import random
import gym
import gym_open_poker
import pandas as pd


def get_winning_rate(win_list, N_count, NN_count):
    print(
        f"Non-novelty tournament Winning rate = {round(np.mean(win_list[:NN_count]), 2)} +- {round(np.std(win_list[:NN_count]), 2)}"
    )
    if N_count != 0:
        print(
            f"Novelty tournament Winning rate = {round(np.mean(win_list[NN_count:]), 2)} +- {round(np.std(win_list[NN_count:]), 2)}"
        )

    return round(np.mean(win_list), 2), round(np.std(win_list), 2)


OUTPUT_PATH = "./agents/results/"
TRAIN_TOURNAMENT = 300
TOTAL_AGENT_COUNT = 9


def training(config_dict, param, MODEL_NAME, random_seed=15, lr=None, delta=None):

    CASH_PLOT_NAME = MODEL_NAME
    MSE_PLOT_NAME = MODEL_NAME

    N_ratio = 0

    # MODEL_NAME = f'dqn_3p3r3d_w_dropout_{TRAIN_TOURNAMENT}'
    # CASH_PLOT_NAME = f'dqn_3p3r3d_w_dropout_cash_plot_{TRAIN_TOURNAMENT}.png'
    # MSE_PLOT_NAME = f'dqn_3p3r3d_w_dropout_mse_plot_{TRAIN_TOURNAMENT}.png'

    NN_count = TRAIN_TOURNAMENT // 2
    N_count = TRAIN_TOURNAMENT // 2

    novelty_detection_list = [None] * (NN_count + N_count)
    # summary
    win_list = [None] * (NN_count + N_count)
    game_hist = [None] * (NN_count + N_count)

    """
    # agent
    if lr is None:
        owe_agent = AgentDQN(storing_folder_name="train_" + MODEL_NAME)
    else:
        owe_agent = AgentDQN(storing_folder_name="train_" + MODEL_NAME, alpha=lr)
    """
    owe_agent = AgentDQNReplayFrame(storing_folder_name="train_" + MODEL_NAME, alpha=lr, delta=delta)

    if os.path.isfile(OUTPUT_PATH + MODEL_NAME + ".keras"):
        read_dqn_path = OUTPUT_PATH + MODEL_NAME + ".keras"
    else:
        read_dqn_path = None

    owe_agent.initialize(learning=True, read_dqn_path=read_dqn_path)

    # training

    # create environment

    novelty_list = config_dict["novelty_list"]

    output_path = "./train_" + MODEL_NAME + "/" + param + "/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    random.seed(random_seed)
    num_list = list(range(100000))
    random.shuffle(num_list)
    rand_seed_list = num_list[: (NN_count + N_count)]
    # print(rand_seed_list)

    # novelty_gt_list
    novel_tournament_count = int(N_ratio * N_count)
    NN_novelty_inject_list = [1] * novel_tournament_count + [0] * (N_count - novel_tournament_count)
    random.shuffle(NN_novelty_inject_list)
    novelty_gt_list = [0] * NN_count + NN_novelty_inject_list

    for tournament_idx in range(NN_count + N_count):
        print("tournament " + str(tournament_idx + 1))
        # modify log_file_path in config_dict to ./timestamp/tournamentX.log
        config_dict["log_file_path"] = output_path + "tournament" + str(tournament_idx + 1) + ".log"

        if novelty_gt_list[tournament_idx] == 1:
            # novelty list inject
            for novelty_string in novelty_list:
                module = __import__("gym_open_poker.wrappers", fromlist=["object"])
                novelty = getattr(module, novelty_string, None)
                if novelty is None:
                    print("Novelty " + novelty_string + "is not found, please check")
                    raise
                env = novelty(gym.make("gym_open_poker/OpenPoker-v0", **config_dict))
                # logger.debug('Injecting the novelty: ' + novelty_string)
        else:
            env = gym.make("gym_open_poker/OpenPoker-v0", **config_dict)
        # train
        owe_agent.train(env, seed=rand_seed_list[tournament_idx])

        # summary
        # novelty detection
        novelty_detected = owe_agent.novelty_detection()
        novelty_detection_list[tournament_idx] = novelty_detected

        # game_hist
        game_hist[tournament_idx] = env.get_tournament_summary()
        # get max_game_index
        max_game_num = max(game_hist[tournament_idx]["cash"].keys())
        if game_hist[tournament_idx]["cash"][max_game_num][0] == max(game_hist[tournament_idx]["cash"][max_game_num]):
            win_list[tournament_idx] = 1
        else:
            win_list[tournament_idx] = 0

    # win_list, game_summary = get_summary()
    # print(win_list)
    # print(game_summary[0])
    # print(game_summary[3])
    win_mean, win_std = get_winning_rate(win_list, N_count, NN_count)
    # print('------')
    # print('confusion_matrix')
    # owe.get_detection_confusion_matrix(output_format='percentage')

    owe_agent.save_model(MODEL_NAME)
    owe_agent.get_cash_graph(param + "_T" + str(TRAIN_TOURNAMENT), episode_line=False)
    owe_agent.get_mse_graph(param + "_T" + str(TRAIN_TOURNAMENT), episode_line=False)
    owe_agent.get_cash_graph(param + "_T" + str(TRAIN_TOURNAMENT), episode_line=True)
    owe_agent.get_mse_graph(param + "_T" + str(TRAIN_TOURNAMENT), episode_line=True)
    # owe.get_detection_confusion_matrix(output_format='count')
    # owe.get_summary_figure()

    game_summary = game_hist
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

    # statistic, pvalue = scipy.stats.ttest_rel(average_cash_list[:30], average_cash_list[30:])
    # ttest_string = "statistic: " + str(round(statistic, 2)) + ', pvalue = ' + str(round(pvalue, 8))

    plt.errorbar(idx_list, average_cash_list, std_cash_list, ls="none", marker="o")
    # plt.axvline(x=len(game_summary)//2, color='red')
    # plt.gca().invert_yaxis()
    plt.xlabel("Tournament")
    plt.ylabel("Cash")
    # plt.annotate(ttest_string, xy=(0.8, 0.8))
    plt.savefig(output_path + "cash_seed" + str(random_seed) + ".png")
    plt.close()
    return (win_mean, win_std)


def main():
    with open("./config.yaml", "r") as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # single train
    agent_random_num = 0
    agent_dump_num = 0
    agent_p_num = 9
    # for lr in np.arange(0.01, 0.02, 0.01):
    #    lr = round(lr, 2)

    lr = 0.0001
    delta = 0.9
    lr = round(lr, 5)
    win_mean_list = []
    win_std_list = []
    param_list = []
    param = f"r{agent_random_num}_d{agent_dump_num}_p{agent_p_num}"
    config_dict["background_agents_raw_list"] = [
        {"agent_type": "agent_p", "number_of_agent": agent_p_num},
        {"agent_type": "agent_dump", "number_of_agent": agent_dump_num},
        {"agent_type": "agent_random", "number_of_agent": agent_random_num},
    ]

    # model_name = f"0517_{param}_T{TRAIN_TOURNAMENT}_replay_frame_lr{lr}_delta{delta}"
    model_name = f"0628_{param}_T{TRAIN_TOURNAMENT}_replay_frame_lr{lr}_delta{delta}"
    print(model_name)
    win_mean, win_std = training(config_dict, param, model_name, lr=lr, delta=delta)
    param_list.append(param)
    win_mean_list.append(win_mean)
    win_std_list.append(win_std)
    output_df = pd.DataFrame({"param": param_list, "win_mean_list": win_mean_list, "win_std_list": win_std_list})
    output_df.to_csv(
        f"{OUTPUT_PATH}/train_{model_name}/train_{model_name}_sum.csv",
        mode="a",
        header=False,
    )
    


if __name__ == "__main__":
    main()
