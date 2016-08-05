require 'nn'

require 'evaluateBoard'
require 'drawBoard'

------------teach tic tac toe----------------
function challengeO(net, player, draw)

    --set optionals
    draw = draw or false

    --set turn variable
    local turn = 1

    --initialize outcomes
    local xWin = false
    local oWin = false
    local tie = false

    --initial board states
    local xState = torch.zeros(18)
    local oState = torch.zeros(18)
    local Q = torch.zeros(9)

    local normQ = nn.SoftMax()

    if draw then drawBoard(xState) end

    --loop until game ends
    while true do

        --X turn (human or random)
        if player then
            repeat
                local action = io.read();
                if xState[action]==0 and xState[9+action]==0 then
                    xState[action] = 1
                    break	--break inner loop
                end
            until false
        else
            local xSort = torch.randperm(9)
            for xLoop=1,9 do
                --repeat until valid move
                if xState[xSort[xLoop]]==0 and xState[9+xSort[xLoop]]==0 then
                    xState[xSort[oLoop]] = 1
                    break	--break inner loop
                end
            end
        end

        --evaluate state
        xWin,oWin,tie = evaluateBoard(xState)
        if draw then drawBoard(xState) end

        --finish game if terminal
        if xWin or oWin or tie then
            break
        end

        --O turn
        oState[{{1,9}}] = xState[{{10,18}}]
        oState[{{10,18}}] = xState[{{1,9}}]
        if player then
            Q = normQ:forward(net:forward(oState)*1e2)
            local spin = torch.uniform()
            for oLoop=1,9 do
                --repeat until valid move
                if spin<Q[{{1,oLoop}}]:sum() then
                    if oState[oLoop]==0 and oState[oLoop+9]==0 then
                        xState[oLoop+9] = 1
                        if draw then print(Q[oLoop]*100) end
                        break	--break inner loop
                    end
                end
            end
        else
            Q = net:forward(oState)
            _,Qsort = torch.sort(Q,1,true)
            for oLoop=1,9 do
                --repeat until valid move
                if xState[Qsort[oLoop]]==0 and xState[Qsort[oLoop]+9]==0 then
                    xState[Qsort[oLoop]+9] = 1
                    break   --break inner loop
                end
            end            
        end


        --evaluate state
        xWin,oWin,tie = evaluateBoard(xState)
        if draw then drawBoard(xState) end

        --finish game if terminal condition
        if xWin or oWin or tie then
            break
        end

        turn = turn + 1

    end
    
    if xWin then
        return 1
    elseif tie then
        return 0
    else
        return -1
    end
    
end

