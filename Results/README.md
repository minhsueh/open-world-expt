DQN training script:
&nbsp;./DQN/train_agent_q_replay_memory_frame.py

Figure 3(a)
experiment script:
&nbsp;./DQN/test_agent_q_replay_memory_frame.py
plotting script:
&nbsp;./DQN/dqn_performance_seperate.py
results:
&nbsp;./DQN/agents/results/0517_dqn_agent_testing/NN_seed20/0517_r0_d0_p9_T300_replay_frame_lr0.0001_delta0.9/0517_r0_d0_p9_T300_replay_frame_lr0.0001_delta0.9_NN_seed20_avg_cash.png
---<br>
Figure 3(b), (c), (d)
experiment script:
&nbsp;./depth1_novely/test_agent_q_replay_memory_frame_batch.sh
plotting script:
&nbsp;./depth1_novely/pool_estimate_pre_post_all_nov.py
results:
&nbsp;./depth1_novely/agents/results/0520_dqn_agent_testing/cash_agent_dump.png
&nbsp;./depth1_novely/agents/results/0520_dqn_agent_testing/cash_agent_p.png
&nbsp;./depth1_novely/agents/results/0520_dqn_agent_testing/cash_agent_random.png

---<br>
Figure 4
experiment script:
&nbsp;same as Figure 3(b), (c), (d)
plotting script:
&nbsp;./depth1_novely/action_cash.py
results:
&nbsp;./depth1_novely/agent_dump_action.png
&nbsp;./depth1_novely/agent_p_action.png
&nbsp;./depth1_novely/agent_random_action.png

other sub-figures corresponded to player_1_normalized_action.png and player_2_normalized_action.png in ./depth1_novely/0520_dqn_agent_testing/action.GameFoldRestrict/0517_r0_d0_p9_T300_replay_frame_lr0.0001_delta0.9/
---<br>
Figure 5 & S5
experiment script:
&nbsp;./depth1_novely2/test_agent_q_replay_memory_frame_batch_depth2.sh
plotting script:
&nbsp;./depth1_novely2/non_linear_heat.py
results:
&nbsp;./depth2_novely/agents/results/0524_dqn_agent_testing/heatcash_agent_dump.png
&nbsp;./depth2_novely/agents/results/0524_dqn_agent_testing/heatcash_agent_p.png
&nbsp;./depth2_novely/agents/results/0524_dqn_agent_testing/heatcash_agent_random.png

---<br>
Figure 6 & S6
experimenal script:
&nbsp;./depth_expt/test_agent_q_replay_memory_frame_batch_random.sh
plotting script:
&nbsp;./depth_expt/novelty_d.py
results:
&nbsp;./depth_expt/agents/results/depth_DQN_againstAgentD.png
&nbsp;./depth_expt/agents/results/depth_DQN_againstAgentP.png
&nbsp;./depth_expt/agents/results/depth_DQN_againstAgentR.png


---<br>
Figure S1
Each sub-figure corresponds to the sub-directory in ./DQN/LR_tuning.

---<br>
Figure S2
experimenal script:
&nbsp;./DQN/test_agent_q_replay_memory_frame_0626.py
&nbsp;./DQN/test_agent_q_replay_memory_frame_0625.py
plotting script:
&nbsp;./DQN/dqn_performance_0626.py
results:
&nbsp;./DQN/agents/results/0626_dqn_agent_testing/NN/0517_r0_d0_p9_T300_replay_frame_lr0.0001_delta0.9/0517_r0_d0_p9_T300_replay_frame_lr0.0001_delta0.9_NN_avg_cash.png

---<br>
Figure S3 & S4
experimenal script:
&nbsp;./mixed_back_agents/test_agent_q_replay_memory_frame_batch.sh
plotting script:
&nbsp;Null, plotting procedure was executed along with the experimental scripts
results:
&nbsp;./mixed_back_agents/0627_dqn_agent_testing/action.GameFoldRestrict/ (Figure S3)
&nbsp;./mixed_back_agents/0627_dqn_agent_testing/card.CardDistHigh/ (Figure S4)