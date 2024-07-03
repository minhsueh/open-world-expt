# declare -a arr=((9 0 0) (0 0 9));


novelty_array=(
    card.CardDistHigh
    card.CardDistLow
    card.CardDistSuit
    card.CardDistOdd
    card.CardDistColor
    conclude_game.Incentive
    conclude_game.Tipping
    conclude_game.SeatChanging
    conclude_game.LuckySeven
    conclude_game.HiddenCard
    game_element.AllOdd
    game_element.BigBlindChange
    game_element.BigBetChange
    game_element.BuyIn
    game_element.TournamentLength
    agent.AgentExchange
    agent.AddAgentR
    agent.AddAgentConservative
    agent.AddAgentAggressive
    agent.AddAgentStirring
    action.GameFoldRestrict
    action.NoFreeLunch
    action.ActionHierarchy
    action.WealthTax
    action.RoundActionReStrict
)


novelty_array=(
    card.CardDistLow
    card.CardDistSuit
    card.CardDistOdd
    card.CardDistColor
    conclude_game.Incentive
    conclude_game.Tipping
    conclude_game.SeatChanging
    conclude_game.LuckySeven
    conclude_game.HiddenCard
    game_element.AllOdd
    game_element.BigBlindChange
    game_element.BigBetChange
    game_element.BuyIn
    game_element.TournamentLength
    agent.AgentExchange
    agent.AddAgentR
    agent.AddAgentConservative
    agent.AddAgentAggressive
    agent.AddAgentStirring
    action.NoFreeLunch
    action.ActionHierarchy
    action.WealthTax
    action.RoundActionReStrict
)

agent_array=(
    -0-0-7
)



IFS='.'

for((ni = 0; ni < ${#novelty_array[@]}; ni++))
do
    
    novelty=${novelty_array[${ni}]}
    # echo $novelty
    for((ai = 0; ai < ${#agent_array[@]}; ai++))
    do
        agent_param=${agent_array[${ai}]}
        # echo "${novelty}${agent_param}"
        python3 test_agent_q_replay_memory_frame_batch.py "${novelty}${agent_param}"
    done

done



# python3 test_agent_q_novelty.py CardDistHigh 9 0 0