require 'nn'

require 'evaluateBoard'
require 'drawBoard'

------------teach tic tac toe----------------
function challenge(net, player, draw)

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

    --loop until game ends
    while true do

        --X turn
        if player then
            Q = normQ:forward(net:forward(xState)*1e2)
            local spin = torch.uniform()
            for xLoop=1,9 do
                --repeat until valid move
                if spin<Q[{{1,xLoop}}]:sum() then
                    if xState[xLoop]==0 and xState[xLoop+9]==0 then
                        xState[xLoop] = 1
                        if draw then print(Q[xLoop]*100) end
                        break	--break inner loop
                    end
                end
            end
        else
            Q = net:forward(xState)
            _,Qsort = torch.sort(Q,1,true)
            for xLoop=1,9 do
                --repeat until valid move
                if xState[Qsort[xLoop]]==0 and xState[9+Qsort[xLoop]]==0 then
                    xState[Qsort[xLoop]] = 1
                    break   --break inner loop
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
        if player then
            repeat
                local action = io.read();
                if xState[action]==0 and xState[9+action]==0 then
                    xState[9+action] = 1
                    break	--break inner loop
                end
            until false
        else
            local oSort = torch.randperm(9)
            for oLoop=1,9 do
                --repeat until valid move
                if xState[oSort[oLoop]]==0 and xState[9+oSort[oLoop]]==0 then
                    xState[9+oSort[oLoop]] = 1
                    break	--break inner loop
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

