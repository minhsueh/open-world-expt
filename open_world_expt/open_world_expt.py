"""
This module is one level higher than open_poker; in other words, we utilize open_poker to test the AI agent's novelty detection and accommodation. 

Specifically, this experiment has two portions, including first non_novle(NN), and followed by novel(N) tournaments. Note that each
tournament call open_poker once, where novel tournaments utilize wrapper to inject novelty.

1. The first NN_count tournaments are original poker without any novelties.
2. The next N_count tournaments contain (N_ratio * N_count) novel tournaments and ((1 - N_ratio) * N_count) non_novelty tournaments

For example,
If we set NN_count = N_count = 30, and N_ratio = 0.6:
1. The first 30 tournaments are original poker without any novelties.
2. The next 30 tournaments contain 18 novel and 12 non_novelty tournaments. Note that we select novel tournaments randomly with N_ratio.



Hyperparameters:
NN_count(int), default = 30
N_count(int), default = 30
N_ratio(float), this value is bounded between 0 to 1. default = 0.6


"""

import gym
import gym_open_poker
from gym_open_poker.envs.poker_util.novelty_generator import NoveltyGenerator
import yaml
import random
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
import collections
import scipy
import json
import importlib


try:
    from importlib import resources as impresources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as impresources


class OpenWorldExpt:
    def __init__(self, owe_agent, NN_count=30, N_count=30, N_ratio=0.6, random_seed=15):
        """
        PARAMS:
            owe_agent()
            NN_count(int): the first starting tournaments number for non_novelty
            N_count(int): the latter tournaments number
            N_ratio(float): the ratio for novel tournament in N_count
        """
        assert isinstance(NN_count, int)
        assert isinstance(N_count, int)
        assert 0 <= N_ratio <= 1

        self.NN_count = NN_count
        self.N_count = N_count
        self.N_ratio = N_ratio
        self.random_seed = random_seed

        self.novelty_detection_list = [None] * (NN_count + N_count)

        # summary
        self.win_list = [None] * (NN_count + N_count)
        self.tournament_hist = [None] * (NN_count + N_count)

        self.executed = False

        # owe_user
        self.owe_agent = owe_agent

        self.folder_name = None

    def execute(self, folder_name=None, config_path=None, config_dict=None):
        """
        The main experiment
        PARAMS:
            config_path

        """
        # load config parameters
        if config_dict:
            pass

        elif config_path:
            with open(config_path, "r") as stream:
                try:
                    config_dict = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
        else:
            from . import default_setting

            try:
                deafault_file = impresources.files(default_setting) / "default_config.yaml"
                with open(deafault_file, "r") as stream:
                    try:
                        config_dict = yaml.safe_load(stream)
                    except yaml.YAMLError as exc:
                        print(exc)
            except AttributeError:
                print("Please check Python version >= 3.7")
                raise

        novelty_list = config_dict["novelty_list"]

        if folder_name:
            self.folder_name = folder_name
            self.output_path = "./" + folder_name + "/"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
        else:
            expt_create_time = str(round(time.time()))
            self.output_path = "./" + expt_create_time + "/"
            if not os.path.exists(expt_create_time):
                os.makedirs(expt_create_time)

        # random seed list
        random.seed(self.random_seed)
        num_list = list(range(100000))
        random.shuffle(num_list)
        rand_seed_list = num_list[: (self.NN_count + self.N_count)]

        # novelty_gt_list
        novel_tournament_count = int(self.N_ratio * self.N_count)
        NN_novelty_inject_list = [1] * novel_tournament_count + [0] * (self.N_count - novel_tournament_count)
        random.shuffle(NN_novelty_inject_list)
        self.novelty_gt_list = [0] * self.NN_count + NN_novelty_inject_list
        print(self.novelty_gt_list)
        ##
        for tournament_idx in range(self.NN_count + self.N_count):
            print("tournament " + str(tournament_idx + 1))
            # modify log_file_path in config_dict to ./timestamp/tournamentX.log
            config_dict["log_file_path"] = self.output_path + "/tournament" + str(tournament_idx + 1) + ".log"
            if self.novelty_gt_list[tournament_idx] == 1:
                self.env = gym.make("gym_open_poker/OpenPoker-v0", **config_dict)
                ng = NoveltyGenerator()
                self.env = ng.inject(self.env, config_dict["novelty_list"])
                """
                # novelty list inject
                for novelty_string in novelty_list:
                    module = __import__("gym_open_poker.wrappers", fromlist=["object"])
                    novelty = getattr(module, novelty_string, None)
                    if novelty is None:
                        print("Novelty " + novelty_string + " is not found, please check")
                        raise
                    self.env = novelty(gym.make("gym_open_poker/OpenPoker-v0", **config_dict))
                    # logger.debug('Injecting the novelty: ' + novelty_string)
                """
            else:
                self.env = gym.make("gym_open_poker/OpenPoker-v0", **config_dict)

            observation, info = self.env.reset(seed=rand_seed_list[tournament_idx])
            reward, terminated, truncated = 0, False, False
            # print('============================')
            # print('---observation---')
            # print(observation)
            # print('---info---')
            # print(info)
            while True:
                # print('============================')
                # print('Enter your action:')

                # keyborad
                # user_action = input()

                # random
                # action_mask = info['action_masks'].astype(bool)
                # all_action_list = np.array(list(range(6)))
                # user_action = np.random.choice(all_action_list[action_mask], size=1).item()

                # OweAgent

                user_action = self.owe_agent.action(observation, reward, terminated, truncated, info)

                if int(user_action) not in range(6):
                    print("It is not a valid action, current value = " + user_action)
                    continue
                # print('----------------')
                observation, reward, terminated, truncated, info = self.env.step(int(user_action))
                # print('---observation---')
                # print(observation)
                # print('---reward---')
                # print(reward)
                # print('---info---')
                # print(info)
                if truncated or terminated:
                    # self.owe_agent.action(observation, reward, terminated, truncated, info) # for rl agent final observation
                    break

            # novelty detection
            novelty_detected = self.owe_agent.novelty_detection()
            self.novelty_detection_list[tournament_idx] = novelty_detected

            # tournament_hist
            self.tournament_hist[tournament_idx] = self.env.get_tournament_summary()
            # get max_game_index
            max_game_num = max(self.tournament_hist[tournament_idx]["cash"].keys())
            if self.tournament_hist[tournament_idx]["cash"][max_game_num][0] == max(
                self.tournament_hist[tournament_idx]["cash"][max_game_num]
            ):
                self.win_list[tournament_idx] = 1
            else:
                self.win_list[tournament_idx] = 0
        self.executed = True

    def get_summary(self):
        if not self.executed:
            print("OpenWorldExpt has not executed yet, please call execute first")
            raise
        return (self.win_list, self.tournament_hist)

    def get_winning_rate(self):
        if not self.executed:
            print("OpenWorldExpt has not executed yet, please call execute first")
            raise
        print(
            f"Non-novelty tournament Winning rate = {round(np.mean(self.win_list[:self.NN_count]), 2)} += {round(np.std(self.win_list[:self.NN_count]), 2)})"
        )
        if self.N_count != 0:
            print(
                f"Novelty tournament Winning rate = {round(np.mean(self.win_list[self.NN_count:]), 2)} += {round(np.std(self.win_list[self.NN_count:]), 2)}"
            )

        return round(np.mean(self.win_list), 2), round(np.std(self.win_list), 2)
        # return round(np.mean(self.win_list[:self.NN_count]), 2), round(np.mean(self.win_list[self.NN_count:]), 2)
        # else:
        #     return round(np.mean(self.win_list[:self.NN_count]), 2)

    def get_detection_confusion_matrix(self, output_format="percentage"):
        """
        Args:
            output_format(str): percentage or count
        """
        if not self.executed:
            print("OpenWorldExpt has not executed yet, please call execute first")
            raise
        assert output_format in ["percentage", "count"]

        tp, tn, fp, fn = self._get_confusion_matrix(
            self.novelty_gt_list[self.NN_count :], self.novelty_detection_list[self.NN_count :]
        )

        leng = len(self.novelty_gt_list[self.NN_count :])

        if output_format == "percentage":
            print("tp percentage = " + str(round(tp / leng, 2)))
            print("tn percentage = " + str(round(tn / leng, 2)))
            print("fp percentage  = " + str(round(fp / leng, 2)))
            print("fn percentage = " + str(round(fn / leng, 2)))
        elif output_format == "count":
            print("tp count = " + str(tp))
            print("tn count = " + str(tn))
            print("fp count  = " + str(fp))
            print("fn count = " + str(fn))
        else:
            raise

    def get_summary_figure(self):
        """
        Output plot that contain each player's cash and rank in each tournament

        """
        # player level
        self.get_player_cash_summary()
        self.get_player_rank_summary()

        # model level
        self.get_model_cash_summary()
        self.get_model_rank_summary()
        self.get_model_win_rate_summary()
        self.get_model_winning_percentage_summary()

    def get_player_cash_summary(self):
        for tournament_idx, tournament_summary in enumerate(self.tournament_hist):
            max_game_num = max(tournament_summary["cash"].keys())
            player_num = len(tournament_summary["cash"][0])
            game_agent_list = tournament_summary["player_strategy"][max_game_num]
            df = pd.DataFrame(
                np.nan,
                index=list(range(max_game_num + 1)),
                columns=["player_" + str(player_idx) for player_idx in range(1, player_num + 1)],
            )
            agent_string_map = {"agent_random": "r", "agent_dump": "d", "agent_p": "p"}

            # cash
            for game_idx in range(max_game_num + 1):

                game_cash_summary = tournament_summary["cash"][game_idx]
                for player_idx in range(player_num):
                    df.loc[game_idx]["player_" + str(player_idx + 1)] = game_cash_summary[player_idx]

            new_column_list = []
            for player_idx in range(player_num):
                if player_idx == 0:
                    new_column_list.append(f"p{player_idx+1}(dqn)")
                else:
                    new_column_list.append(f"p{player_idx+1}({agent_string_map[game_agent_list[player_idx]]})")
            df.columns = new_column_list

            output_plot = df.plot(marker="o")
            plt.grid()
            plt.xticks(range(1, max_game_num + 1))
            plt.xlabel("Game number")
            plt.ylabel("Player's Cash")
            output_plot.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            output_plot.figure.savefig(self.output_path + str(tournament_idx + 1) + ".png")
            plt.clf()
            plt.close()

    def get_player_rank_summary(self):
        pass

    def get_model_win_rate_summary(self):
        """
        Get the detail summary in each tournament.
        Specifically, get the win rate of different strategic agents, including agent_random, agent_dump, agent_p

        RETURNS:
            win_rate_dict(dict of float): indicate the model performance over all tournament, including average win rate and std
        OUTPUT:
            create a errorbar plot with x-axis as tournament number, and y-axis is win rate
        Note:
            example of win_rate_dict: {
                model_random_avg: 0.1,
                model_random_std: 0.1,
            }


        """
        if not self.executed:
            print("OpenWorldExpt has not executed yet, please call execute first")
            raise

        out_dict = {
            "player_1": [],
            "agent_random": [],
            "agent_random_std": [],
            "agent_random_ste": [],
            "agent_dump": [],
            "agent_dump_std": [],
            "agent_dump_ste": [],
            "agent_p": [],
            "agent_p_std": [],
            "agent_p_ste": [],
            "others": [],
            "others_std": [],
            "others_ste": [],
        }

        for tournament_idx, tournament_summary in enumerate(self.tournament_hist):
            tem_dict = {"player_1": [], "agent_random": [], "agent_dump": [], "agent_p": [], "others": []}

            # we only care about the last game in each tournament
            max_game_num = max(tournament_summary["cash"].keys())
            player_num = len(tournament_summary["cash"][0])

            #
            game_agent_list = tournament_summary["player_strategy"][max_game_num]
            game_cash_summary = tournament_summary["cash"][max_game_num]
            winner_cash_amount = max(game_cash_summary)
            # player_1_cash = game_cash_summary[0]
            for player_idx in range(player_num):
                agent = game_agent_list[player_idx]
                if agent in tem_dict:
                    if game_cash_summary[player_idx] == winner_cash_amount:
                        tem_dict[agent].append(1)
                    else:
                        tem_dict[agent].append(0)
                    if player_idx != 0:
                        tem_dict["others"].append(tem_dict[agent][-1])

            # aggregate:
            for model in ["player_1", "agent_random", "agent_dump", "agent_p", "others"]:
                out_dict[model].append(np.mean(tem_dict[model]))
                if model != "player_1":
                    out_dict[model + "_std"].append(np.std(tem_dict[model]))
                    out_dict[model + "_ste"].append(np.std(tem_dict[model], ddof=1) / np.sqrt(np.size(tem_dict[model])))

        data_x = range(1, len(out_dict["player_1"]) + 1)
        out_dict["T"] = data_x

        df = pd.DataFrame(out_dict)

        # aggregation
        retrun_dict = dict()
        for model in ["player_1", "agent_random", "agent_dump", "agent_p", "others"]:
            retrun_dict[model + "_win_rate"] = np.mean(df[model])
            retrun_dict[model + "_win_rate_std"] = np.std(df[model])
            retrun_dict[model + "_win_rate_ste"] = np.std(df[model], ddof=1) / np.sqrt(np.size(df[model]))

        # hypothesis testing
        for model in ["agent_random", "agent_dump", "agent_p", "others"]:
            statistic, p_value = scipy.stats.ttest_ind(df["player_1"], df[model], alternative="greater")
            alpha = 0.05  # Set your significance level
            if p_value < alpha:
                retrun_dict["win_rate_significant_than_" + model] = "*"
            else:
                retrun_dict["win_rate_significant_than_" + model] = ""

        # plotting
        color_list = ["red", "green", "blue", "darkviolet"]
        model_list = ["player_1", "agent_random", "agent_dump", "agent_p"]
        label_map = {"player_1": "dqn", "agent_random": "model_random", "agent_dump": "model_dump", "agent_p": "model_p"}
        figure, ax = plt.subplots()
        for model, color in zip(model_list, color_list):
            if model == "player_1":
                ax.plot(df["T"], df[model], c=color, label=label_map[model], marker="x", ls="None")
            elif not df[model].isnull().values.any():
                markers, caps, bars = ax.errorbar(
                    df["T"],
                    df[model],
                    df[model + "_ste"],
                    c=color,
                    label=label_map[model] + retrun_dict["win_rate_significant_than_" + model],
                    capsize=5,
                    fmt="o",
                )
                [bar.set_alpha(0.3) for bar in bars]
                [cap.set_alpha(0.3) for cap in caps]
        plt.grid()
        # plt.xticks(range(1,max_game_num+1))
        plt.xlabel("Tournament number")
        plt.ylabel("Winning rate")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        ax.figure.savefig(self.output_path + "model_avg_win_rate.png")
        plt.clf()
        plt.close()

        df.to_csv(self.output_path + "model_avg_win_rate.csv")

        return retrun_dict

    def get_model_cash_summary(self):
        """
        Get the detail cash summary in each tournament.
        Specifically, get the cash of different strategic agents, including agent_random, agent_dump, agent_p

        """
        if not self.executed:
            print("OpenWorldExpt has not executed yet, please call execute first")
            raise

        out_dict = {
            "player_1": [],
            "agent_random": [],
            "agent_random_std": [],
            "agent_random_ste": [],
            "agent_dump": [],
            "agent_dump_std": [],
            "agent_dump_ste": [],
            "agent_p": [],
            "agent_p_std": [],
            "agent_p_ste": [],
            "others": [],
            "others_std": [],
            "others_ste": [],
        }

        for tournament_idx, tournament_summary in enumerate(self.tournament_hist):
            tem_dict = {"player_1": [], "agent_random": [], "agent_dump": [], "agent_p": [], "others": []}
            # we only care about the last game in each tournament
            max_game_num = max(tournament_summary["cash"].keys())
            player_num = len(tournament_summary["cash"][0])

            #
            game_agent_list = tournament_summary["player_strategy"][max_game_num]
            game_cash_summary = tournament_summary["cash"][max_game_num]
            # player_1_cash = game_cash_summary[0]
            for player_idx in range(player_num):
                agent = game_agent_list[player_idx]
                if agent in tem_dict:
                    tem_dict[agent].append(game_cash_summary[player_idx])
                    if player_idx != 0:
                        tem_dict["others"].append(game_cash_summary[player_idx])

            # aggregate:
            for model in ["player_1", "agent_random", "agent_dump", "agent_p", "others"]:
                out_dict[model].append(np.mean(tem_dict[model]))
                if model != "player_1":
                    out_dict[model + "_std"].append(np.std(tem_dict[model]))
                    out_dict[model + "_ste"].append(np.std(tem_dict[model], ddof=1) / np.sqrt(np.size(tem_dict[model])))

        data_x = range(1, len(out_dict["player_1"]) + 1)
        out_dict["T"] = data_x

        df = pd.DataFrame(out_dict)

        # aggregation
        retrun_dict = dict()
        for model in ["player_1", "agent_random", "agent_dump", "agent_p", "others"]:
            retrun_dict[model + "_cash"] = np.mean(df[model])
            retrun_dict[model + "_cash_std"] = np.std(df[model])
            retrun_dict[model + "_cash_ste"] = np.std(df[model], ddof=1) / np.sqrt(np.size(df[model]))

        # hypothesis testing
        for model in ["agent_random", "agent_dump", "agent_p", "others"]:
            statistic, p_value = scipy.stats.ttest_ind(df["player_1"], df[model], alternative="greater")
            alpha = 0.05  # Set your significance level
            if p_value < alpha:
                retrun_dict["cash_significant_than_" + model] = "*"
            else:
                retrun_dict["cash_significant_than_" + model] = ""

        # plotting
        color_list = ["red", "green", "blue", "darkviolet"]
        model_list = ["player_1", "agent_random", "agent_dump", "agent_p"]
        label_map = {"player_1": "dqn", "agent_random": "model_random", "agent_dump": "model_dump", "agent_p": "model_p"}
        figure, ax = plt.subplots()
        for model, color in zip(model_list, color_list):
            if model == "player_1":
                ax.plot(df["T"], df[model], c=color, label=label_map[model], marker="x", ls="None")
            elif not df[model].isnull().values.any():
                markers, caps, bars = ax.errorbar(
                    df["T"],
                    df[model],
                    df[model + "_ste"],
                    c=color,
                    label=label_map[model] + retrun_dict["cash_significant_than_" + model],
                    capsize=5,
                    fmt="o",
                )
                [bar.set_alpha(0.3) for bar in bars]
                [cap.set_alpha(0.3) for cap in caps]
        plt.grid()
        # plt.xticks(range(1,max_game_num+1))
        plt.xlabel("Tournament number")
        plt.ylabel("Average cash")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        ax.figure.savefig(self.output_path + "model_avg_cash.png")
        plt.clf()
        plt.close()

        df.to_csv(self.output_path + "model_avg_cash.csv")

        return retrun_dict

    def get_model_rank_summary(self):
        """
        Get the detail cash summary in each tournament.
        Specifically, get the cash of different strategic agents, including agent_random, agent_dump, agent_p

        """
        if not self.executed:
            print("OpenWorldExpt has not executed yet, please call execute first")
            raise

        out_dict = {
            "player_1": [],
            "agent_random": [],
            "agent_random_std": [],
            "agent_random_ste": [],
            "agent_dump": [],
            "agent_dump_std": [],
            "agent_dump_ste": [],
            "agent_p": [],
            "agent_p_std": [],
            "agent_p_ste": [],
            "others": [],
            "others_std": [],
            "others_ste": [],
        }

        for tournament_idx, tournament_summary in enumerate(self.tournament_hist):
            tem_dict = {"player_1": [], "agent_random": [], "agent_dump": [], "agent_p": [], "others": []}

            # we only care about the last game in each tournament
            max_game_num = max(tournament_summary["rank"].keys())
            player_num = len(tournament_summary["rank"][max_game_num])

            #
            game_agent_list = tournament_summary["player_strategy"][max_game_num]
            game_rank_summary = tournament_summary["final_rank_list"]
            for player_idx in range(player_num):
                agent = game_agent_list[player_idx]
                if agent in tem_dict:
                    tem_dict[agent].append(game_rank_summary[player_idx])
                    if player_idx != 0:
                        tem_dict["others"].append(game_rank_summary[player_idx])

            # aggregate:
            for model in ["player_1", "agent_random", "agent_dump", "agent_p", "others"]:
                out_dict[model].append(np.mean(tem_dict[model]))
                if model != "player_1":
                    out_dict[model + "_std"].append(np.std(tem_dict[model]))
                    out_dict[model + "_ste"].append(np.std(tem_dict[model], ddof=1) / np.sqrt(np.size(tem_dict[model])))

        data_x = range(1, len(out_dict["player_1"]) + 1)
        out_dict["T"] = data_x

        df = pd.DataFrame(out_dict)

        # aggregation
        retrun_dict = dict()
        for model in ["player_1", "agent_random", "agent_dump", "agent_p", "others"]:
            retrun_dict[model + "_rank"] = np.mean(df[model])
            retrun_dict[model + "_rank_std"] = np.std(df[model])
            retrun_dict[model + "_rank_ste"] = np.std(df[model], ddof=1) / np.sqrt(np.size(df[model]))

        # hypothesis testing
        for model in ["agent_random", "agent_dump", "agent_p", "others"]:
            statistic, p_value = scipy.stats.ttest_ind(df["player_1"], df[model], alternative="greater")
            alpha = 0.05  # Set your significance level
            if p_value < alpha:
                retrun_dict["rank_significant_than_" + model] = "*"
            else:
                retrun_dict["rank_significant_than_" + model] = ""

        # plotting
        color_list = ["red", "green", "blue", "darkviolet"]
        model_list = ["player_1", "agent_random", "agent_dump", "agent_p"]
        label_map = {"player_1": "dqn", "agent_random": "model_random", "agent_dump": "model_dump", "agent_p": "model_p"}
        figure, ax = plt.subplots()
        for model, color in zip(model_list, color_list):
            if model == "player_1":
                ax.plot(df["T"], df[model], c=color, label=label_map[model], marker="x", ls="None")
            elif not df[model].isnull().values.any():
                markers, caps, bars = ax.errorbar(
                    df["T"],
                    df[model],
                    df[model + "_ste"],
                    c=color,
                    label=label_map[model] + retrun_dict["rank_significant_than_" + model],
                    capsize=5,
                    fmt="o",
                )
                [bar.set_alpha(0.3) for bar in bars]
                [cap.set_alpha(0.3) for cap in caps]
        plt.grid()
        # plt.xticks(range(1,max_game_num+1))
        plt.xlabel("Tournament number")
        plt.ylabel("Average rank")
        plt.ylim(-1, 11)
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.gca().invert_yaxis()
        ax.figure.savefig(self.output_path + "model_avg_rank.png")
        plt.clf()
        plt.close()

        df.to_csv(self.output_path + "model_avg_rank.csv")

        return retrun_dict

    def get_model_winning_percentage_summary(self):
        """
        Get the winning percentage of each player in each tournament.
        Winning percentage is defined as the proportion of players a participant has beaten, expressed as a percentage.


        Specifically, get the winning percentag of different strategic agents, including agent_random, agent_dump, agent_p

        """
        if not self.executed:
            print("OpenWorldExpt has not executed yet, please call execute first")
            raise

        out_dict = {
            "player_1": [],
            "agent_random": [],
            "agent_random_std": [],
            "agent_random_ste": [],
            "agent_dump": [],
            "agent_dump_std": [],
            "agent_dump_ste": [],
            "agent_p": [],
            "agent_p_std": [],
            "agent_p_ste": [],
            "others": [],
            "others_std": [],
            "others_ste": [],
        }

        for tournament_idx, tournament_summary in enumerate(self.tournament_hist):
            tem_dict = {"player_1": [], "agent_random": [], "agent_dump": [], "agent_p": [], "others": []}

            # we only care about the last game in each tournament
            max_game_num = max(tournament_summary["cash"].keys())
            player_num = len(tournament_summary["cash"][0])

            #
            game_agent_list = tournament_summary["player_strategy"][max_game_num]
            game_cash_summary = tournament_summary["cash"][max_game_num]

            cash_counter = collections.Counter(game_cash_summary)
            sort_game_cash_summary = sorted(cash_counter.keys())
            win_counter = dict()
            pre = 0

            for cash in sort_game_cash_summary:
                win_counter[cash] = pre / (player_num - 1)
                pre += cash_counter[cash]

            # player_1_cash = game_cash_summary[0]
            for player_idx in range(player_num):
                agent = game_agent_list[player_idx]
                if agent in tem_dict:
                    tem_dict[agent].append(win_counter[game_cash_summary[player_idx]])
                    if player_idx != 0:
                        tem_dict["others"].append(win_counter[game_cash_summary[player_idx]])

            # aggregate:
            for model in ["player_1", "agent_random", "agent_dump", "agent_p", "others"]:
                out_dict[model].append(np.mean(tem_dict[model]) * 100)
                if model != "player_1":
                    out_dict[model + "_std"].append(np.std(tem_dict[model]) * 100)
                    out_dict[model + "_ste"].append(np.std(tem_dict[model], ddof=1) / np.sqrt(np.size(tem_dict[model])) * 100)

        data_x = range(1, len(out_dict["player_1"]) + 1)
        out_dict["T"] = data_x

        df = pd.DataFrame(out_dict)

        # aggregation
        retrun_dict = dict()
        for model in ["player_1", "agent_random", "agent_dump", "agent_p", "others"]:
            retrun_dict[model + "_winning_percentage"] = np.mean(df[model])
            retrun_dict[model + "_winning_percentage_std"] = np.std(df[model])
            retrun_dict[model + "_winning_percentage_ste"] = np.std(df[model], ddof=1) / np.sqrt(np.size(df[model]))

        # hypothesis testing
        for model in ["agent_random", "agent_dump", "agent_p", "others"]:
            statistic, p_value = scipy.stats.ttest_ind(df["player_1"], df[model], alternative="greater")
            alpha = 0.05  # Set your significance level
            if p_value < alpha:
                retrun_dict["winning_percentage_significant_than_" + model] = "*"
            else:
                retrun_dict["winning_percentage_significant_than_" + model] = ""

        # plotting
        color_list = ["red", "green", "blue", "darkviolet"]
        model_list = ["player_1", "agent_random", "agent_dump", "agent_p"]
        label_map = {"player_1": "dqn", "agent_random": "model_random", "agent_dump": "model_dump", "agent_p": "model_p"}
        figure, ax = plt.subplots()
        for model, color in zip(model_list, color_list):
            if model == "player_1":
                ax.plot(df["T"], df[model], c=color, label=label_map[model], marker="x", ls="None")
            elif not df[model].isnull().values.any():
                markers, caps, bars = ax.errorbar(
                    df["T"],
                    df[model],
                    df[model + "_ste"],
                    c=color,
                    label=label_map[model] + retrun_dict["winning_percentage_significant_than_" + model],
                    capsize=5,
                    fmt="o",
                )
                [bar.set_alpha(0.3) for bar in bars]
                [cap.set_alpha(0.3) for cap in caps]
        plt.grid()
        # plt.xticks(range(1,max_game_num+1))
        plt.xlabel("Tournament number")
        plt.ylabel("Average win percentage(%)")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        ax.figure.savefig(self.output_path + "model_avg_win_percentage.png")
        plt.clf()
        plt.close()

        df.to_csv(self.output_path + "model_avg_win_percentage.csv")
        return retrun_dict

    def _get_confusion_matrix(self, gt, prediction):
        """
        Args:
            gt(list): 1 or 0
            prediction(list): 1 or 0
        """
        assert len(gt) == len(prediction)

        tp, tn, fp, fn = 0, 0, 0, 0
        for idx in range(len(gt)):
            if prediction[idx] == 1:
                if gt[idx] == 1:
                    tp += 1
                elif gt[idx] == 0:
                    fp += 1
            elif prediction[idx] == 0:
                if gt[idx] == 1:
                    fn += 1
                elif gt[idx] == 0:
                    tn += 1
            else:
                raise

        return (tp, tn, fp, fn)

    def get_player_action_profile_summary(self):
        """
        get agent's action count.
        For example, within all tournaments, how many fold did player_1 conduct

        OUTPUTS:
            output_dict: {
                tournament_1: {
                    player_1: {
                        strategy: str
                        fold: int
                        bet: int
                        ...
                    },
                    player_2: {

                    }
                },
                tournament_2: {
                    player_1: {
                        strategy: str
                        fold: int
                        bet: int
                        ...
                    },
                    player_2: {

                    }
                },
                ...
                overall: {
                    player_1: {
                        strategy: str
                        fold: int
                        bet: int
                        ...
                    },
                    player_2: {

                    }
                }
            }


        """
        if not self.executed:
            print("OpenWorldExpt has not executed yet, please call execute first")
            raise

        output_dict = dict()
        overall_dict = None

        action_list = ["FOLD", "CHECK", "CALL", "BET", "RAISE_BET", "ALL_IN"]
        model_abbreviation_dict = {"player_1": "dqn", "agent_random": "r", "agent_dump": "d", "agent_p": "p"}
        action_count_dict = dict()
        total_tournament_count = len(self.tournament_hist)

        for tournament_idx, tournament_summary in enumerate(self.tournament_hist):
            tem_dict = {"player_1": [], "agent_random": [], "agent_dump": [], "agent_p": [], "others": []}

            # we only care about the last game in each tournament
            max_game_num = max(tournament_summary["rank"].keys())
            player_num = len(tournament_summary["cash"][0])

            #
            game_agent_list = tournament_summary["player_strategy"][max_game_num]

            tem_dict = dict()
            for player_idx in range(1, player_num + 1):
                tem_dict["player_" + str(player_idx)] = dict()
                for a in action_list:
                    tem_dict["player_" + str(player_idx)][a] = [0] * max_game_num
            for player_idx in range(1, player_num + 1):
                tem_dict["player_" + str(player_idx)]["model"] = game_agent_list[player_idx - 1]

            action_history = tournament_summary["action_history"]
            max_game_num = max(tournament_summary["action_history"].keys())

            for game_idx in range(1, max_game_num + 1):
                for player_idx in range(1, player_num + 1):
                    for a in action_history[game_idx]["player_" + str(player_idx)]:
                        tem_dict["player_" + str(player_idx)][a][game_idx - 1] += 1
            output_dict[tournament_idx] = tem_dict
            """
            # action count
            action_count_dict = dict()
            for tournament_idx in range(len(self.tournament_hist)):
                action_count_dict[tournament_idx] = dict()
                for player_idx in range(1, player_num + 1):
                    player_name = "player_" + str(player_idx)
                    action_count_dict[tournament_idx][player_name] = {a: [] for a in action_list}
                    for a in action_list:
                        action_count_dict[tournament_idx][player_name][a].append(
                            output_dict[tournament_idx][player_name][a] if a in output_dict[tournament_idx][player_name] else 0
                        )
            """

        output_dict["overall"] = overall_dict
        with open(self.output_path + "action_summary.json", "w") as outfile:
            json.dump(output_dict, outfile)

        player_num = len(self.tournament_hist[0]["cash"][0])
        # aggregation
        aggregate_dict = dict()
        tournament_list = range(1, len(self.tournament_hist) + 1)
        aggregate_dict["Tournament"] = tournament_list
        for player_idx in range(1, player_num + 1):
            player_name = "player_" + str(player_idx)
            aggregate_dict[player_name + "_cash"] = []
            # cash
            for tournament_idx, tournament_summary in enumerate(self.tournament_hist):
                max_game_num = max(tournament_summary["cash"].keys())
                game_cash_summary = tournament_summary["cash"][max_game_num]

                aggregate_dict[player_name + "_cash"].append(game_cash_summary[player_idx - 1])

            # action
            for a in action_list:
                aggregate_dict[player_name + "_" + a + "_mean"] = []
                aggregate_dict[player_name + "_" + a + "_ste"] = []
                aggregate_dict[player_name + "_" + a + "_sample_size"] = []
                for tournament_idx, tournament_summary in enumerate(self.tournament_hist):
                    aggregate_dict[player_name + "_" + a + "_mean"].append(
                        np.mean(output_dict[tournament_idx][player_name][a])
                    )
                    aggregate_dict[player_name + "_" + a + "_ste"].append(
                        np.std(output_dict[tournament_idx][player_name][a], ddof=1)
                        / np.sqrt(np.size(output_dict[tournament_idx][player_name][a]))
                    )
                    aggregate_dict[player_name + "_" + a + "_sample_size"].append(
                        np.size(output_dict[tournament_idx][player_name][a])
                    )

        aggregate_df = pd.DataFrame(aggregate_dict)
        aggregate_df.to_csv(self.output_path + "aggregation.csv")
        # plot player wise
        for player_idx in range(1, player_num + 1):
            player_name = "player_" + str(player_idx)
            fig, ax = plt.subplots()
            action_color_list = ["black", "darkviolet", "blue", "green", "gold", "red"]
            for a, color in zip(action_list, action_color_list):
                ax.errorbar(
                    aggregate_df["Tournament"],
                    aggregate_df[player_name + "_" + a + "_mean"],
                    aggregate_df[player_name + "_" + a + "_ste"],
                    c=color,
                    label=a,
                    capsize=5,
                    fmt="o",
                )
            plt.grid()
            tem_model = model_abbreviation_dict[game_agent_list[player_idx - 1]]
            plt.title(f"player_{player_idx}({tem_model})")
            plt.xlabel("Tournament number")
            plt.xlim(-5, total_tournament_count + 5)
            ax.set_xticks(range(0, total_tournament_count + 20, 20))
            plt.ylabel("Action count")
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.tight_layout()
            fig.savefig(self.output_path + player_name + "_action.png")

        for player_idx in range(1, player_num + 1):
            player_name = "player_" + str(player_idx)
            fig, ax = plt.subplots()
            ax.scatter(aggregate_df["Tournament"], aggregate_df[player_name + "_cash"])

            tem_model = model_abbreviation_dict[game_agent_list[player_idx - 1]]
            plt.grid()
            plt.title(f"player_{player_idx}({tem_model})")
            plt.xlabel("Tournament number")
            plt.xlim(-5, total_tournament_count + 5)
            ax.set_xticks(range(0, total_tournament_count + 20, 20))
            plt.ylabel("Cash")
            plt.ylim(-100, 2100)
            # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.tight_layout()
            fig.savefig(self.output_path + player_name + "_cash.png")

        return output_dict

    def get_player_action_percentage_profile_summary(self):
        """
        get agent's action count.
        For example, within all tournaments, how many fold did player_1 conduct

        OUTPUTS:
            output_dict: {
                tournament_1: {
                    player_1: {
                        strategy: str
                        fold: int
                        bet: int
                        ...
                    },
                    player_2: {

                    }
                },
                tournament_2: {
                    player_1: {
                        strategy: str
                        fold: int
                        bet: int
                        ...
                    },
                    player_2: {

                    }
                },
                ...
                overall: {
                    player_1: {
                        strategy: str
                        fold: int
                        bet: int
                        ...
                    },
                    player_2: {

                    }
                }
            }


        """
        if not self.executed:
            print("OpenWorldExpt has not executed yet, please call execute first")
            raise

        output_dict = dict()
        overall_dict = None

        action_list = ["FOLD", "CHECK", "CALL", "BET", "RAISE_BET", "ALL_IN"]
        model_abbreviation_dict = {"player_1": "dqn", "agent_random": "r", "agent_dump": "d", "agent_p": "p"}
        action_count_dict = dict()
        total_tournament_count = len(self.tournament_hist)
        # count the action from raw data
        """
        output_dict = {
            tournament_idx: {
                player_name: {
                    fold: [1, 0, ....] # size = total game in tournamet
                    ...
                }
            }
        }
        """
        for tournament_idx, tournament_summary in enumerate(self.tournament_hist):
            tem_dict = {"player_1": [], "agent_random": [], "agent_dump": [], "agent_p": [], "others": []}

            # we only care about the last game in each tournament
            max_game_num = max(tournament_summary["rank"].keys())
            # player_num = len(tournament_summary["rank"][0])

            player_num = 0

            #
            game_agent_list = tournament_summary["player_strategy"][max_game_num]
            for tem_agent in game_agent_list:
                if tem_agent in tem_dict:
                    player_num += 1

            tem_dict = dict()
            for player_idx in range(1, player_num + 1):
                tem_dict["player_" + str(player_idx)] = dict()
                for a in action_list:
                    tem_dict["player_" + str(player_idx)][a] = [0] * max_game_num
            for player_idx in range(1, player_num + 1):
                tem_dict["player_" + str(player_idx)]["model"] = game_agent_list[player_idx - 1]

            action_history = tournament_summary["action_history"]
            max_game_num = max(tournament_summary["action_history"].keys())

            for game_idx in range(1, max_game_num + 1):
                for player_idx in range(1, player_num + 1):
                    for a in action_history[game_idx]["player_" + str(player_idx)]:
                        tem_dict["player_" + str(player_idx)][a][game_idx - 1] += 1
            output_dict[tournament_idx] = tem_dict

        output_dict["overall"] = overall_dict
        with open(self.output_path + "action_summary.json", "w") as outfile:
            json.dump(output_dict, outfile)

        # aggregation
        player_num = len(self.tournament_hist[0]["cash"][0])
        aggregate_dict = dict()
        tournament_list = range(1, len(self.tournament_hist) + 1)
        aggregate_dict["Tournament"] = tournament_list
        total_action_count_dict = dict()
        for player_idx in range(1, player_num + 1):
            player_name = "player_" + str(player_idx)
            aggregate_dict[player_name + "_cash"] = []
            total_action_count_dict[player_name] = []
            # cash
            for tournament_idx, tournament_summary in enumerate(self.tournament_hist):
                max_game_num = max(tournament_summary["cash"].keys())
                game_cash_summary = tournament_summary["cash"][max_game_num]
                aggregate_dict[player_name + "_cash"].append(game_cash_summary[player_idx - 1])

                # tem_action_count
                tem_action_count_list = []

                for game_idx in range(len(output_dict[tournament_idx][player_name][a])):
                    tem_action_count = 0
                    for a in action_list:
                        tem_action_count += output_dict[tournament_idx][player_name][a][game_idx]
                    tem_action_count_list.append(tem_action_count)
                total_action_count_dict[player_name].append(tem_action_count_list)

            # action
            for a in action_list:
                aggregate_dict[player_name + "_" + a + "_normalized_mean"] = []
                aggregate_dict[player_name + "_" + a + "_normalized_ste"] = []
                aggregate_dict[player_name + "_" + a + "_sample_size"] = []
                for tournament_idx, tournament_summary in enumerate(self.tournament_hist):

                    action_count_list = np.array(output_dict[tournament_idx][player_name][a])
                    action_total_count_list = np.array(total_action_count_dict[player_name][tournament_idx])
                    assert len(action_count_list) == len(action_total_count_list)
                    normalized_action_count_list = (
                        np.divide(
                            action_count_list,
                            action_total_count_list,
                            out=np.zeros(action_count_list.shape, dtype=float),
                            where=action_total_count_list != 0,
                        )
                        * 100
                    )

                    aggregate_dict[player_name + "_" + a + "_normalized_mean"].append(np.mean(normalized_action_count_list))
                    aggregate_dict[player_name + "_" + a + "_normalized_ste"].append(
                        np.std(normalized_action_count_list, ddof=1) / np.sqrt(np.size(normalized_action_count_list))
                    )
                    aggregate_dict[player_name + "_" + a + "_sample_size"].append(np.size(normalized_action_count_list))

        aggregate_df = pd.DataFrame(aggregate_dict)
        aggregate_df.to_csv(self.output_path + "normalized_aggregation.csv")
        # plot player wise
        for player_idx in range(1, player_num + 1):
            player_name = "player_" + str(player_idx)
            fig, ax = plt.subplots()
            action_color_list = ["black", "darkviolet", "blue", "green", "gold", "red"]
            for a, color in zip(action_list, action_color_list):
                ax.errorbar(
                    aggregate_df["Tournament"],
                    aggregate_df[player_name + "_" + a + "_normalized_mean"],
                    aggregate_df[player_name + "_" + a + "_normalized_ste"],
                    c=color,
                    label=a,
                    capsize=5,
                    fmt="o",
                )
            plt.grid()
            tem_model = model_abbreviation_dict[game_agent_list[player_idx - 1]]
            plt.title(f"player_{player_idx}({tem_model})")
            plt.xlabel("Tournament number")
            plt.xlim(-5, total_tournament_count + 5)
            ax.set_xticks(range(0, total_tournament_count + 20, 20))
            plt.ylabel("Action count percentage")
            plt.ylim(-5, 105)
            ax.set_yticks(range(0, 120, 20))
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.tight_layout()
            fig.savefig(self.output_path + player_name + "_normalized_action.png")

        for player_idx in range(1, player_num + 1):
            player_name = "player_" + str(player_idx)
            fig, ax = plt.subplots()
            ax.scatter(aggregate_df["Tournament"], aggregate_df[player_name + "_cash"])

            tem_model = model_abbreviation_dict[game_agent_list[player_idx - 1]]
            plt.grid()
            plt.title(f"player_{player_idx}({tem_model})")
            plt.xlabel("Tournament number")
            plt.xlim(-5, total_tournament_count + 5)
            ax.set_xticks(range(0, total_tournament_count + 20, 20))
            plt.ylabel("Cash")
            plt.ylim(-100, 2100)
            # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.tight_layout()
            fig.savefig(self.output_path + player_name + "_cash.png")

        return output_dict

    def get_pre_post_model_summary(self, measurement):
        assert measurement in ["cash", "rank", "win_rate", "winning_percentage"]
        if not self.executed:
            print("OpenWorldExpt has not executed yet, please call execute first")
            raise

        out_dict = {
            "pre_player_1": [],
            "post_player_1": [],
            "pre_agent_random": [],
            "pre_agent_random_std": [],
            "pre_agent_random_ste": [],
            "post_agent_random": [],
            "post_agent_random_std": [],
            "post_agent_random_ste": [],
            "pre_agent_dump": [],
            "pre_agent_dump_std": [],
            "pre_agent_dump_ste": [],
            "post_agent_dump": [],
            "post_agent_dump_std": [],
            "post_agent_dump_ste": [],
            "pre_agent_p": [],
            "pre_agent_p_std": [],
            "pre_agent_p_ste": [],
            "post_agent_p": [],
            "post_agent_p_std": [],
            "post_agent_p_ste": [],
            "pre_others": [],
            "pre_others_std": [],
            "pre_others_ste": [],
            "post_others": [],
            "post_others_std": [],
            "post_others_ste": [],
        }

        for tournament_idx, tournament_summary in enumerate(self.tournament_hist):
            tem_dict = {"player_1": [], "agent_random": [], "agent_dump": [], "agent_p": [], "others": []}

            if measurement == "cash":
                # we only care about the last game in each tournament
                max_game_num = max(tournament_summary["cash"].keys())
                player_num = len(tournament_summary["cash"][0])

                #
                game_agent_list = tournament_summary["player_strategy"][max_game_num]
                game_cash_summary = tournament_summary["cash"][max_game_num]
                # player_1_cash = game_cash_summary[0]
                for player_idx in range(player_num):
                    agent = game_agent_list[player_idx]
                    if agent in tem_dict:
                        tem_dict[agent].append(game_cash_summary[player_idx])
                        if player_idx != 0:
                            tem_dict["others"].append(game_cash_summary[player_idx])
            elif measurement == "rank":
                # we only care about the last game in each tournament
                max_game_num = max(tournament_summary["rank"].keys())
                player_num = len(tournament_summary["rank"][0])

                #
                game_agent_list = tournament_summary["player_strategy"][max_game_num]
                game_rank_summary = tournament_summary["final_rank_list"]
                for player_idx in range(player_num):
                    agent = game_agent_list[player_idx]
                    if agent in tem_dict:
                        tem_dict[agent].append(game_rank_summary[player_idx])
                        if player_idx != 0:
                            tem_dict["others"].append(game_rank_summary[player_idx])

            elif measurement == "win_rate":
                # we only care about the last game in each tournament
                max_game_num = max(tournament_summary["cash"].keys())
                player_num = len(tournament_summary["cash"][0])

                #
                game_agent_list = tournament_summary["player_strategy"][max_game_num]
                game_cash_summary = tournament_summary["cash"][max_game_num]
                winner_cash_amount = max(game_cash_summary)
                # player_1_cash = game_cash_summary[0]
                for player_idx in range(player_num):
                    agent = game_agent_list[player_idx]
                    if agent in tem_dict:
                        if game_cash_summary[player_idx] == winner_cash_amount:
                            tem_dict[agent].append(1)
                        else:
                            tem_dict[agent].append(0)
                        if player_idx != 0:
                            tem_dict["others"].append(tem_dict[agent][-1])
            elif measurement == "winning_percentage":
                # we only care about the last game in each tournament
                max_game_num = max(tournament_summary["cash"].keys())
                player_num = len(tournament_summary["cash"][0])

                #
                game_agent_list = tournament_summary["player_strategy"][max_game_num]
                game_cash_summary = tournament_summary["cash"][max_game_num]

                cash_counter = collections.Counter(game_cash_summary)
                sort_game_cash_summary = sorted(cash_counter.keys())
                win_counter = dict()
                pre = 0

                for cash in sort_game_cash_summary:
                    win_counter[cash] = pre / (player_num - 1)
                    pre += cash_counter[cash]

                # player_1_cash = game_cash_summary[0]
                for player_idx in range(player_num):
                    agent = game_agent_list[player_idx]
                    if agent in tem_dict:
                        tem_dict[agent].append(win_counter[game_cash_summary[player_idx]])
                        if player_idx != 0:
                            tem_dict["others"].append(win_counter[game_cash_summary[player_idx]])

            # aggregate:
            for model in ["player_1", "agent_random", "agent_dump", "agent_p", "others"]:
                coef = 1
                if measurement == "winning_percentage":
                    coef = 100

                out_dict["pre_" + model].append(np.mean(tem_dict[model]) * coef)
                if model != "player_1":
                    out_dict["pre_" + model + "_std"].append(np.std(tem_dict[model], ddof=1) * coef)
                    out_dict["pre_" + model + "_ste"].append(
                        np.std(tem_dict[model], ddof=1) / np.sqrt(np.size(tem_dict[model])) * coef
                    )

                out_dict["post_" + model].append(np.mean(tem_dict[model]) * coef)
                if model != "player_1":
                    out_dict["post_" + model + "_std"].append(np.std(tem_dict[model], ddof=1) * coef)
                    out_dict["post_" + model + "_ste"].append(
                        np.std(tem_dict[model], ddof=1) / np.sqrt(np.size(tem_dict[model])) * coef
                    )

        # print(out_dict)
        df = pd.DataFrame(out_dict)
        df.to_csv(self.output_path + "pre_post_" + measurement + ".csv")
        # aggregation

        self.NN_count = 20
        """
        retrun_dict = dict()
        for model in ["player_1", "agent_random", "agent_dump", "agent_p", "others"]:
            retrun_dict["pre_" + model + "_" + measurement] = np.mean(df["pre_" + model][: self.NN_count])
            retrun_dict["pre_" + model + "_" + measurement + "_std"] = np.std(df["pre_" + model][: self.NN_count], ddof=1)
            retrun_dict["pre_" + model + "_" + measurement + "_ste"] = np.std(
                df["pre_" + model][: self.NN_count], ddof=1
            ) / np.sqrt(np.size(df["pre_" + model][: self.NN_count]))

            retrun_dict["post_" + model + "_" + measurement] = np.mean(df["post_" + model][self.NN_count :])
            retrun_dict["post_" + model + "_" + measurement + "_std"] = np.std(df["post_" + model][self.NN_count :], ddof=1)
            retrun_dict["post_" + model + "_" + measurement + "_ste"] = np.std(
                df["post_" + model][self.NN_count :], ddof=1
            ) / np.sqrt(np.size(df["post_" + model][self.NN_count :]))
        """
        retrun_dict = dict()
        for model in ["player_1", "agent_random", "agent_dump", "agent_p", "others"]:
            retrun_dict["pre_" + model + "_" + measurement] = np.mean(df["pre_" + model][: self.NN_count])
            if model == "player_1":
                retrun_dict["pre_" + model + "_" + measurement + "_std"] = np.std(df["pre_" + model][: self.NN_count], ddof=1)
                retrun_dict["pre_" + model + "_" + measurement + "_ste"] = np.std(
                    df["pre_" + model][: self.NN_count], ddof=1
                ) / np.sqrt(np.size(df["pre_" + model][: self.NN_count]))
            else:
                tem_std = np.array(df["pre_" + model + "_std"][: self.NN_count])
                retrun_dict["pre_" + model + "_" + measurement + "_std"] = (sum(tem_std * tem_std) / len(tem_std)) ** 0.5
                retrun_dict["pre_" + model + "_" + measurement + "_ste"] = (sum(tem_std * tem_std)) ** 0.5 / len(tem_std)

            retrun_dict["post_" + model + "_" + measurement] = np.mean(df["post_" + model][self.NN_count :])
            if model == "player_1":
                retrun_dict["post_" + model + "_" + measurement + "_std"] = np.std(df["post_" + model][self.NN_count :], ddof=1)
                retrun_dict["post_" + model + "_" + measurement + "_ste"] = np.std(
                    df["post_" + model][self.NN_count :], ddof=1
                ) / np.sqrt(np.size(df["post_" + model][self.NN_count :]))
            else:
                tem_std = np.array(df["post_" + model + "_std"][self.NN_count :]) 
                retrun_dict["post_" + model + "_" + measurement + "_std"] = (sum(tem_std * tem_std) / len(tem_std)) ** 0.5
                retrun_dict["post_" + model + "_" + measurement + "_ste"] = (sum(tem_std * tem_std)) ** 0.5 / len(tem_std)
        return retrun_dict
