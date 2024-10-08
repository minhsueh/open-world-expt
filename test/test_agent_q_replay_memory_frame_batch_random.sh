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
    conclude_game.Tipping
    game_element.BigBetChange
)




agent_array=(
    -7-0-0
    -0-7-0
    -0-0-7
)




IFS='.'

n2='action.NoFreeLunch'

for ((ti = 1; ti < 11; ti++))
do
    seed=$(($RANDOM%100))
    for((ni = 1; ni < 6; ni++))
    do
        for((ai = 0; ai < ${#agent_array[@]}; ai++))
        do  
            agent_param=${agent_array[${ai}]}
            echo "${seed}${agent_param}-${ni}"
            python3 test_agent_q_replay_memory_frame_batch_random.py "${seed}${agent_param}-${ni}-${ti}"
        done
        
    done
done


: '
for((ni = 0; ni < ${#novelty_array[@]}; ni++))
do
    for((mi = $ni; mi < ${#novelty_array[@]}; mi++))
    do
        n1=${novelty_array[${ni}]}
        n2=${novelty_array[${mi}]}

        nc1="$( cut -d '.' -f 1 <<< "$n1" )"
        nc2="$( cut -d '.' -f 1 <<< "$n2" )"

        if [[ "$nc1" != "$nc2" ]]
        then
            novelty=${novelty_array[${ni}]}+${novelty_array[${mi}]}
            # echo $novelty
            for((ai = 0; ai < ${#agent_array[@]}; ai++))
            do
                agent_param=${agent_array[${ai}]}
                # echo "${novelty}${agent_param}"
                python3 test_agent_q_replay_memory_frame_batch.py "${novelty}${agent_param}"
            done
        fi
    done
done
'


# python3 test_agent_q_novelty.py CardDistHigh 9 0 0