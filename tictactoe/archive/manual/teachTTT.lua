--dependencies
require 'torch'
require 'nn'
require 'learn'
require 'evaluateBoard'
require 'drawBoard'
--require 'cudatorch'	--eventually, maybe

------------teach tic tac toe----------------
function teachTTT(net, eps)

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
                    _,besti = torch.max(Q,1)
                    p1Action = besti[1]
                else
                    i = torch.randperm(9)
                    besti = i[1]
                    p1Action = besti
                end
                --punish for invalid moves
                if xState[besti] ~= 0 then
                    net = learn(net, p1State, p1State, p1Action, invalidScore, false)
                    break	--break inner loop
                end
                --update p1 state
                xState[besti] = 1
                newp1State = xState:clone()
                done = true	--exit outer loop
            until true
        until done

        --update p2 state
        newp2State = -xState

        --evaluate state
        xWin,oWin,tie = evaluateBoard(xState)
        --drawBoard(xState)

        --p1 self-eval
        if xWin then		--win
            net = learn(net, p1State, newp1State, p1Action, winScore, true)
        elseif tie then		--tie
            net = learn(net, p1State, newp1State, p1Action, tieScore, true)
        end

        --p2 opponent eval
        if turn > 1 then
            if xWin then	--lose
                net = learn(net, p2State, newp2State, p2Action, loseScore, true)
            elseif tie then	--tie
                net = learn(net, p2State, newp2State, p2Action, tieScore, true)
            else		--uneventful
                net = learn(net, p2State, newp2State, p2Action, noScore, false)
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
                if torch.uniform() > eps then
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
                    net = learn(net, p2State, p2State, p2Action, invalidScore, false)
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
        --drawBoard(xState)

        --p2 self-eval
        if oWin then	--win
            net = learn(net, p2State, newp2State, p2Action, winScore, true)
        elseif tie then	--tie
            net = learn(net, p2State, newp2State, p2Action, tieScore, true)
        end

        --p1 opponent eval
        if oWin then	--lose
            net = learn(net, p1State, newp1State, p1Action, loseScore, true)
        elseif tie then	--tie
            net = learn(net, p1State, newp1State, p1Action, tieScore, true)
        else		--uneventful
            net = learn(net, p1State, newp1State, p1Action, noScore, false)
        end

        --finish game if terminal condition
        if xWin or oWin or tie then
            break
        end

        turn = turn + 1

    end
    
    return net

end

