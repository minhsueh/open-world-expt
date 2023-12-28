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
import open_poker
import yaml
import random
from open_poker.wrappers import * # CardDistHigh, CardDistLow
import numpy as np
import time
import os
 


try:
    from importlib import resources as impresources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as impresources

class OpenWorldExpt():
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

        self.novelty_detection_list = [None] * (NN_count+N_count)

        # summary 
        self.win_list = [None] * (NN_count+N_count)
        self.game_hist = [None] * (NN_count+N_count)


        self.executed = False

        # owe_user
        self.owe_agent = owe_agent

    def execute(self, config_path=None):
        """
        The main experiment
        PARAMS:
            config_path
        
        """
        # load config parameters
        if config_path:
            with open(config_path, "r") as stream:
                try:
                    config_dict = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
        else:
            from . import default_setting
            try:
                deafault_file = (impresources.files(default_setting) / 'default_config.yaml')
                with open(deafault_file, "r") as stream:
                    try:
                        config_dict = yaml.safe_load(stream)
                    except yaml.YAMLError as exc:
                        print(exc)
            except AttributeError:
                print('Please check Python version >= 3.7')
                raise

        novelty_list = config_dict['novelty_list']

        expt_create_time = str(round(time.time()))
        self.output_path = './' + expt_create_time + '/'
        if not os.path.exists(expt_create_time):
            os.makedirs(expt_create_time)


        # random seed list
        random.seed(self.random_seed)
        num_list = list(range(100000))
        random.shuffle(num_list)
        rand_seed_list = num_list[:(self.NN_count + self.N_count)]

        ## novelty_gt_list
        novel_tournament_count = int(self.N_ratio * self.N_count)
        NN_novelty_inject_list = [1] * novel_tournament_count + [0] * (self.N_count - novel_tournament_count)
        random.shuffle(NN_novelty_inject_list)
        self.novelty_gt_list = [0] * self.NN_count + NN_novelty_inject_list



        ## 
        for tournament_idx in range(self.NN_count+self.N_count):
            print('tournament ' + str(tournament_idx+1))
            # modify log_file_path in config_dict to ./timestamp/tournamentX.log
            config_dict['log_file_path'] = './' + expt_create_time + '/tournament' + str(tournament_idx+1) + '.log'

            if self.novelty_gt_list[tournament_idx] == 1:
                # novelty list inject
                for novelty_string in novelty_list:
                    module = __import__('open_poker.wrappers', fromlist=['object'])
                    novelty = getattr(module, novelty_string, None)
                    if novelty is None:
                        print('Novelty ' + novelty_string + 'is not found, please check')
                        raise
                    env = novelty(gym.make("open_poker/OpenPoker-v0", **config_dict))
                    logger.debug('Injecting the novelty: ' + novelty_string)
            else:
                env = gym.make("open_poker/OpenPoker-v0", **config_dict)

            observation, info = env.reset(seed=rand_seed_list[tournament_idx])
            reward, terminated, truncated = 0, False, False
            #print('============================')
            #print('---observation---')
            #print(observation)
            #print('---info---')
            #print(info)
            while(True):
               #print('============================')
                #print('Enter your action:')

                # keyborad
                # user_action = input()

                # random
                # action_mask = info['action_masks'].astype(bool)
                # all_action_list = np.array(list(range(6)))
                # user_action = np.random.choice(all_action_list[action_mask], size=1).item()

                # OweAgent

                user_action = self.owe_agent.action(observation, reward, terminated, truncated, info)

                if int(user_action) not in range(6):
                    print('It is not a valid action, current value = ' + user_action)
                    continue
                #print('----------------')
                observation, reward, terminated, truncated, info = env.step(int(user_action))
                #print('---observation---')
                #print(observation)
                #print('---reward---')
                #print(reward)
                #print('---info---')
                #print(info)
                if truncated or terminated:
                    break

            # novelty detection
            novelty_detected = self.owe_agent.novelty_detection()
            self.novelty_detection_list[tournament_idx] = novelty_detected

            # game_hist
            self.game_hist[tournament_idx] = env.get_tournament_summary()
            if self.game_hist[tournament_idx]['rank'][0] == 1:
                self.win_list[tournament_idx] = 1
            else:
                self.win_list[tournament_idx] = 0
        self.executed = True


    def get_summary(self):
        if not self.executed:
            print('OpenWorldExpt has not executed yet, please call execute first')
            raise
        return(self.win_list, self.game_hist)

    def get_winning_rate(self):
        if not self.executed:
            print('OpenWorldExpt has not executed yet, please call execute first')
            raise
        print('Non-novelty tournament Winning rate = ' + str(round(np.mean(self.win_list[:self.NN_count]), 2)))
        print('Novelty tournament Winning rate = ' + str(round(np.mean(self.win_list[self.NN_count:]), 2)))

        return round(np.mean(self.win_list[:self.NN_count]), 2), round(np.mean(self.win_list[self.NN_count:]), 2)

    def get_detection_confusion_matrix(self, output_format='percentage'):
        """
        Args:
            output_format(str): percentage or count
        """
        if not self.executed:
            print('OpenWorldExpt has not executed yet, please call execute first')
            raise
        assert output_format in ['percentage', 'count']

        tp, tn, fp, fn = self._get_confusion_matrix(self.novelty_gt_list[self.NN_count:], self.novelty_detection_list[self.NN_count:])

        leng = len(self.novelty_gt_list[self.NN_count:])

        if output_format == 'percentage':
            print('tp percentage = ' + str(round(tp/leng, 2)))
            print('tn percentage = ' + str(round(tn/leng, 2)))
            print('fp percentage  = ' + str(round(fp/leng, 2)))
            print('fn percentage = ' + str(round(fn/leng, 2)))
        elif output_format == 'count':
            print('tp count = ' + str(tp))
            print('tn count = ' + str(tn))
            print('fp count  = ' + str(fp))
            print('fn count = ' + str(fn))
        else:
            raise

        


    def get_summary_figure(self):
        """
        Output plot that contain each player's cash and rank in each tournament

        """
        import matplotlib.pyplot as plt
        import pandas as pd
        from matplotlib.ticker import MaxNLocator

        for tournament_idx, tournament_summary in enumerate(self.game_hist):
            max_game_num = max(tournament_summary['cash'].keys())
            player_num = len(tournament_summary['cash'][0])
            df = pd.DataFrame(np.nan, index=list(range(max_game_num+1)), columns=['player_'+str(player_idx) for player_idx in range(1,player_num+1)])



            # cash
            for game_idx in range(max_game_num+1):


                game_cash_summary = tournament_summary['cash'][game_idx]
                for player_idx in range(player_num):
                    df.loc[game_idx]['player_'+str(player_idx+1)] = game_cash_summary[player_idx]
            

            output_plot = df.plot(marker='o')
            plt.grid()
            plt.xticks(range(1,max_game_num+1))
            plt.xlabel('Game number')
            plt.ylabel("Player's Cash")
            output_plot.xaxis.set_major_locator(MaxNLocator(integer=True))
            output_plot.figure.savefig(self.output_path + str(tournament_idx+1) + '.png')
            plt.cla()
            plt.close()
            # rank


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

        return(tp, tn, fp, fn)



        











