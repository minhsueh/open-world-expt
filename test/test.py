from open_world_expt import OpenWorldExpt



owe = OpenWorldExpt()
owe.execute()


win_list, game_summary = owe.get_summary()
print(win_list)
print(game_summary[0])
print(game_summary[3])
owe.get_summary_figure()
