from open_world_expt import OpenWorldExpt
from open_world_expt.owe_agent import OweAgent
import numpy as np

class RandomOweAgent(OweAgent):

    def action(self, observation, reward, terminated, truncated, info):
        print('Please input your action:')
        user_action = input()

        return(user_action) 
    def novelty_detection(self):
        print('Please input if you think there is a novelty:')
        detected = input()
        return(detected)



owe_random_agent = RandomOweAgent()

owe = OpenWorldExpt(owe_random_agent)

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
