--dependencies
require 'torch'
require 'nn'
require 'evaluateBoard'
require 'drawBoard'

------------teach tic tac toe----------------
function play(net, dataSet, dataIndex, draw)

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
    p1State = torch.zeros(18):cuda()
    p2State = torch.zeros(18):cuda()
    newp1State = torch.zeros(18):cuda()
    newp2State = torch.zeros(18):cuda()
    xState = torch.zeros(18):cuda()
    Q = torch.zeros(9):cuda()

    --set scores
    winScore = 1
    loseScore = -1
    invalidScore = -1
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
                if xState[p1Action]~=0 or xState[9+p1Action]~=0 then
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
        if draw then drawBoard(xState) end

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
            newp2State[{{10,18}}] = xState[{{10,18}}]
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
        p2State[{{10,18}}] = xState[{{10,18}}]
        done = false
        repeat
            repeat
                --determine next move
                if torch.uniform() > eps then   --always random
                    Q = net:forward(p2State)
                    _,besti = Q:max(1)
                    p2Action = besti[1]
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
                newp2State[{{10,18}}] = xState[{{10,18}}]
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
            dataIndex = dataIndex + 1
            dataSet[dataIndex] = {p2State:clone(), newp2State:clone(), p2Action, winScore, true}
        elseif tie then	--tie
            dataIndex = dataIndex + 1
            dataSet[dataIndex] = {p2State:clone(), newp2State:clone(), p2Action, tieScore, true}
        end

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

        turn = turn + 1

    end
    
    return dataSet, dataIndex

end

