require 'torch'
require 'io'

function drawBoard(board)

    io.write("\t")
    
    if board[1] == 1 then io.write('x')
    elseif board[1] == -1 then io.write('o')
    else io.write('.') end

    io.write("\t|\t")

    if board[2] == 1 then io.write('x')
    elseif board[2] == -1 then io.write('o')
    else io.write('.') end

    io.write("\t|\t")

    if board[3] == 1 then io.write('x')
    elseif board[3] == -1 then io.write('o')
    else io.write('.') end

    io.write("\n\t")

    if board[4] == 1 then io.write('x')
    elseif board[4] == -1 then io.write('o')
    else io.write('.') end

    io.write("\t|\t")

    if board[5] == 1 then io.write('x')
    elseif board[5] == -1 then io.write('o')
    else io.write('.') end

    io.write("\t|\t")

    if board[6] == 1 then io.write('x')
    elseif board[6] == -1 then io.write('o')
    else io.write('.') end

    io.write("\n\t")

    if board[7] == 1 then io.write('x')
    elseif board[7] == -1 then io.write('o')
    else io.write('.') end

    io.write("\t|\t")

    if board[8] == 1 then io.write('x')
    elseif board[8] == -1 then io.write('o')
    else io.write('.') end

    io.write("\t|\t")

    if board[9] == 1 then io.write('x')
    elseif board[9] == -1 then io.write('o')
    else io.write('.') end

    io.write("\n\n\n")

end
