--dependencies
require 'torch'
require 'nn'
require 'evaluateBoard'
require 'drawBoard'

------------teach tic tac toe----------------
function play(net, dataSet, dataIndex)

    --set optionals
    eps = eps or 0.5

    --set turn variable
    turn = 1

    --initial board states
    p1State = torch.zeros(18)
    p2State = torch.zeros(18)
    newp1State = torch.zeros(18)
    newp2State = torch.zeros(18)
    xState = torch.zeros(18)
    Q = torch.zeros(9)

    --set scores
    winScore = 1
    loseScore = -1
    invalidScore = -1
    tieScore = 0
    noScore = 0

    normQ = nn.SoftMax()

    --loop until game ends
    while true do

        --X turn
        p1State = xState:clone()
        done = false
        repeat
            repeat
                --determine next move
                if torch.uniform() > eps then   --exploit
                    --Q = net:forward(p1State)
                    --_,besti = Q:max(1)
                    --p1Action = besti[1]
                    Q = normQ:forward(net:forward(p1State)*1e2)
                    local spin = torch.uniform()
                    for chance = 1,9 do
                        if spin<Q[{{1,chance}}]:sum() then
                            p1Action = chance
                            break
                        end
                    end
                else                            --explore
                    besti = torch.randperm(9)
                    p1Action = besti[1]
                end
                --punish for invalid moves
                if xState[p1Action]~=0 or xState[p1Action+9]~=0 then
                    dataIndex = dataIndex + 1
                    dataSet[dataIndex] = {p1State:clone(), p1State:clone(), p1Action, invalidScore, false}
                    break	--break inner loop
                end
                --update p1 state
                xState[p1Action] = 1
                newp1State = xState:clone()
                done = true	--exit outer loop
            until true
        until done

        --evaluate state
        xWin,oWin,tie = evaluateBoard(xState)

        --p1 self-eval
        if xWin then		--win
            dataIndex = dataIndex + 1
            dataSet[dataIndex] = {p1State:clone(), newp1State:clone(), p1Action, winScore, true}
        elseif tie then		--tie
            dataIndex = dataIndex + 1
            dataSet[dataIndex] = {p1State:clone(), newp1State:clone(), p1Action, tieScore, true}
        end

        --p2 opponent eval
        if turn > 1 then
            --update p2 state
            newp2State[{{1,9}}] = xState[{{10,18}}]
            newp2State[{{10,18}}] = xState[{{1,9}}]
            if xWin then	--lose
                dataIndex = dataIndex + 1
                dataSet[dataIndex] = {p2State:clone(), newp2State:clone(), p2Action, loseScore, true}
            elseif tie then	--tie
                dataIndex = dataIndex + 1
                dataSet[dataIndex] = {p2State:clone(), newp2State:clone(), p2Action, tieScore, true}
            else		--uneventful
                dataIndex = dataIndex + 1
                dataSet[dataIndex] = {p2State:clone(), newp2State:clone(), p2Action, noScore, false}
            end
        end

        --finish game if terminal
        if xWin or oWin or tie then
            break
        end

        --O turn        
        p2State[{{1,9}}] = xState[{{10,18}}]
        p2State[{{10,18}}] = xState[{{1,9}}]
        done = false
        repeat
            repeat
                --determine next move
                if torch.uniform() > eps then
                    --Q = net:forward(p2State)
                    --_,besti = Q:max(1)
                    --p2Action = besti[1]
                    Q = normQ:forward(net:forward(p2State))
                    local spin = torch.uniform()
                    for chance = 1,9 do
                        if spin<Q[{{1,chance}}]:sum() then
                            p2Action = chance
                            break
                        end
                    end
                else
                    besti = torch.randperm(9)
                    p2Action = besti[1]
                end
                --punish for invalid moves
                if xState[p2Action]~=0 or xState[p2Action+9]~=0 then
                    dataIndex = dataIndex + 1
                    dataSet[dataIndex] = {p2State:clone(), p2State:clone(), p2Action, invalidScore, false}
                    break	--break inner loop
                end
                --update p2 state
                xState[p2Action+9] = 1
                newp2State[{{1,9}}] = xState[{{10,18}}]
                newp2State[{{10,18}}] = xState[{{1,9}}]
                done = true	--exit outer loop
            until true
        until done

        --evaluate state
        xWin,oWin,tie = evaluateBoard(xState)

        --p2 self-eval
        if oWin then	--win
            dataIndex = dataIndex + 1
            dataSet[dataIndex] = {p2State:clone(), newp2State:clone(), p2Action, winScore, true}
        elseif tie then	--tie
            dataIndex = dataIndex + 1
            dataSet[dataIndex] = {p2State:clone(), newp2State:clone(), p2Action, tieScore, true}
        end

        --update p1 state
        newp1State = xState:clone()

        --p1 opponent eval
        if oWin then	--lose
            dataIndex = dataIndex + 1
            dataSet[dataIndex] = {p1State:clone(), newp1State:clone(), p1Action, loseScore, true}
        elseif tie then	--tie
            dataIndex = dataIndex + 1
            dataSet[dataIndex] = {p1State:clone(), newp1State:clone(), p1Action, tieScore, true}
        else		--uneventful
            dataIndex = dataIndex + 1
            dataSet[dataIndex] = {p1State:clone(), newp1State:clone(), p1Action, noScore, false}
        end

        --finish game if terminal condition
        if xWin or oWin or tie then
            break
        end

	--nextTurn
        turn = turn + 1

    end
    
    return dataSet, dataIndex

end

