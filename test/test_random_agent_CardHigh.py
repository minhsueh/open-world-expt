from open_world_expt import OpenWorldExpt
from open_world_expt.owe_agent import OweAgent
import numpy as np
import matplotlib.pyplot as plt
import scipy
np.random.seed(10)
class RandomOweAgent(OweAgent):

    def action(self, observation, reward, terminated, truncated, info):
        action_mask = info['action_masks'].astype(bool)
        all_action_list = np.array(list(range(6)))
        user_action = np.random.choice(all_action_list[action_mask], size=1).item()

        return(user_action) 
    def novelty_detection(self):
        return(0)


for random_seed in [15]:
    owe_random_agent = RandomOweAgent()

    owe = OpenWorldExpt(owe_random_agent, N_ratio=1.0, random_seed=random_seed)

    owe.execute()


    win_list, game_summary = owe.get_summary()
    #print(win_list)
    #print(game_summary[0])
    #print(game_summary[3])
    owe.get_winning_rate()
    print('------')
    print('confusion_matrix')
    owe.get_detection_confusion_matrix(output_format='percentage')
    # owe.get_detection_confusion_matrix(output_format='count')
    owe.get_summary_figure()


    # get ranking average
    average_cash_list = []
    std_cash_list = []
    idx_list = []
    for tournament_idx in range(len(game_summary)):
        tem_cash_list = []
        idx_list.append(tournament_idx)
        for game in range(1, max(game_summary[tournament_idx]['cash'].keys())):
            tem_cash_list.append(game_summary[tournament_idx]['cash'][game][0])

        average_cash_list.append(np.mean(tem_cash_list))
        std_cash_list.append(np.std(tem_cash_list))


    statistic, pvalue = scipy.stats.ttest_rel(average_cash_list[:30], average_cash_list[30:])
    ttest_string = "statistic: " + str(round(statistic, 2)) + ', pvalue = ' + str(round(pvalue, 5))

    plt.errorbar(idx_list, average_cash_list, std_cash_list, ls='none', marker='o')
    plt.axvline(x=30, color='red')
    # plt.gca().invert_yaxis()
    plt.xlabel('Tournament')
    plt.ylabel('Cash')
    plt.annotate(ttest_string, xy=(0.8, 0.8))
    plt.savefig('cash_seed' + str(random_seed) + '.png')
    plt.close()
