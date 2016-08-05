require 'torch'
require 'io'

function drawBoard(board)

    xBoard = board[{{1,9}}]
    oBoard = board[{{10,18}}]

    io.write("\t")
    
    if xBoard[1] == 1 then io.write('x')
    elseif oBoard[1] == 1 then io.write('o')
    else io.write('.') end

    io.write("\t|\t")

    if xBoard[2] == 1 then io.write('x')
    elseif oBoard[2] == 1 then io.write('o')
    else io.write('.') end

    io.write("\t|\t")

    if xBoard[3] == 1 then io.write('x')
    elseif oBoard[3] == 1 then io.write('o')
    else io.write('.') end

    io.write("\n\t")

    if xBoard[4] == 1 then io.write('x')
    elseif oBoard[4] == 1 then io.write('o')
    else io.write('.') end

    io.write("\t|\t")

    if xBoard[5] == 1 then io.write('x')
    elseif oBoard[5] == 1 then io.write('o')
    else io.write('.') end

    io.write("\t|\t")

    if xBoard[6] == 1 then io.write('x')
    elseif oBoard[6] == 1 then io.write('o')
    else io.write('.') end

    io.write("\n\t")

    if xBoard[7] == 1 then io.write('x')
    elseif oBoard[7] == 1 then io.write('o')
    else io.write('.') end

    io.write("\t|\t")

    if xBoard[8] == 1 then io.write('x')
    elseif oBoard[8] == 1 then io.write('o')
    else io.write('.') end

    io.write("\t|\t")

    if xBoard[9] == 1 then io.write('x')
    elseif oBoard[9] == 1 then io.write('o')
    else io.write('.') end

    io.write("\n\n\n")

end
