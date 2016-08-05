--dependencies
require 'torch'
require 'nn'
require 'remember'
require 'evaluateBoard'
require 'drawBoard'

------------teach tic tac toe----------------
function play(net, dataset, dataIndex, eps, draw)

    --set optionals
    eps = eps or 0.5
    draw = draw or false

    --set turn variable
    turn = 1

    --initialize outcomes
    xWin = false
    oWin = false
    tie = false

    --initial board states
    p1State = torch.zeros(9)
    p2State = torch.zeros(9)
    newp1State = torch.zeros(9)
    newp2State = torch.zeros(9)
    xState = torch.zeros(9)
    Q = torch.zeros(9)

    --set scores
    winScore = 1
    loseScore = -1
    invalidScore = -2
    tieScore = 0
    noScore = 0

    --loop until game ends
    while true do

        --X turn
        p1State = xState:clone()
        done = false
        repeat
            repeat
                --determine next move
                if torch.uniform() > eps then
                    Q = net:forward(p1State)
                    _,besti = Q:max(1)
                    p1Action = besti[1]
                else
                    i = torch.randperm(9)
                    besti = i[1]
                    p1Action = besti
                end
                --punish for invalid moves
                if xState[p1Action] ~= 0 then
                    dataIndex = dataIndex + 1
                    dataset = remember(net, p1State, p1State, p1Action, invalidScore, false, dataset, dataIndex)
                    break	--break inner loop
                end
                --update p1 state
                xState[p1Action] = 1
                newp1State = xState:clone()
                done = true	--exit outer loop
            until true
        until done

        --update p2 state
        newp2State = -xState

        --evaluate state
        xWin,oWin,tie = evaluateBoard(xState)
        if draw then drawBoard(xState) end

        --p1 self-eval
        if xWin then		--win
            dataIndex = dataIndex + 1
            dataset = remember(net, p1State, newp1State, p1Action, winScore, true, dataset, dataIndex)
        elseif tie then		--tie
            dataIndex = dataIndex + 1
            dataset = remember(net, p1State, newp1State, p1Action, tieScore, true, dataset, dataIndex)
        end

        --p2 opponent eval
        if turn > 1 then
            if xWin then	--lose
                --dataIndex = dataIndex + 1
                --dataset = remember(net, p2State, newp2State, p2Action, loseScore, true, dataset, dataIndex)
            elseif tie then	--tie
                --dataIndex = dataIndex + 1
                --dataset = remember(net, p2State, newp2State, p2Action, tieScore, true, dataset, dataIndex)
            else		--uneventful
                --dataIndex = dataIndex + 1
                --dataset = remember(net, p2State, newp2State, p2Action, noScore, false, dataset, dataIndex)
            end
        end

        --finish game if terminal
        if xWin or oWin or tie then
            break
        end

        --O turn        
        p2State = -xState
        done = false
        repeat
            repeat
                --determine next move
                if torch.uniform() > 1 then   --always random
                    Q = net:forward(p2State)
                    _,besti = torch.max(Q,1)
                    p2Action = besti[1]
                else
                    i = torch.randperm(9)
                    besti = i[1]
                    p2Action = besti
                end
                --punish for invalid moves
                if xState[besti] ~= 0 then
                    --dataIndex = dataIndex + 1
                    --dataset = remember(net, p2State, p2State, p2Action, invalidScore, false, dataset, dataIndex)
                    break	--break inner loop
                end
                --update p2 state
                xState[besti] = -1
                newp2State = -xState
                done = true	--exit outer loop
            until true
        until done

        --update p2 state
        newp1State = xState:clone()

        --evaluate state
        xWin,oWin,tie = evaluateBoard(xState)
        if draw then drawBoard(xState) end

        --p2 self-eval
        if oWin then	--win
            --dataIndex = dataIndex + 1
            --dataset = remember(net, p2State, newp2State, p2Action, winScore, true, dataset, dataIndex)
        elseif tie then	--tie
            --dataIndex = dataIndex + 1
            --dataset = remember(net, p2State, newp2State, p2Action, tieScore, true, dataset, dataIndex)
        end

        --p1 opponent eval
        if oWin then	--lose
            dataIndex = dataIndex + 1
            dataset = remember(net, p1State, newp1State, p1Action, loseScore, true, dataset, dataIndex)
        elseif tie then	--tie
            dataIndex = dataIndex + 1
            dataset = remember(net, p1State, newp1State, p1Action, tieScore, true, dataset, dataIndex)
        else		--uneventful
            dataIndex = dataIndex + 1
            dataset = remember(net, p1State, newp1State, p1Action, noScore, false, dataset, dataIndex)
        end

        --finish game if terminal condition
        if xWin or oWin or tie then
            break
        end

        turn = turn + 1

    end
    
    return dataset, dataIndex

end

