
--dependencies
require "torch"
--require cuda  --maybe

--local class
--localClass = {};

---------evaluate board-----------
function evaluateBoard(board)

    --victory types
    victory = torch.Tensor(8,3);
    victory[1] = torch.Tensor({1, 2, 3})
    victory[2] = torch.Tensor({4, 5, 6})
    victory[3] = torch.Tensor({7, 8, 9})
    victory[4] = torch.Tensor({1, 4, 7})
    victory[5] = torch.Tensor({2, 5, 8})
    victory[6] = torch.Tensor({3, 6, 9})
    victory[7] = torch.Tensor({1, 5, 9})
    victory[8] = torch.Tensor({3, 5, 7})

    --x victory
    gameBoard = board[{{1,9}}]
    for loop = 1,8 do
        xWin = game(loop)
        if xWin then return xWin, oWin, tie end
    end

    --o victory
    gameBoard = board[{{10,18}}]
    for loop = 1,8 do
        oWin = game(loop)
        if oWin then return xWin, oWin, tie end
    end

    --tie game
    moves = board:ne(0):sum()
    tie = moves==9
    if tie then return xWin, oWin, tie end

    --uneventful
    return tie, xWin, oWin
    
end



------helper functions----------

function game(i)
    t = 1;
    for j = 1,3 do
        t = t*gameBoard[victory[i][j]]
    end
    return t ~= 0
end


--localClass.f = f
--return localClass
