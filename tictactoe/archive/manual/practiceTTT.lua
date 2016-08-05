
--dependencies
require 'torch';
require 'cudatorch';


--practice tic tac toe
function teachTTT( net, player, seed )

    --manual seed
    torch.manualSeed(seed);

    --flip coin for X
    firstX = torch.round(torch.random);

    --print move order
    if player then
        if firstX == 1 print("X's!") else print("O's!")
    end

    --set first turn variables
    first = 1;
    turn = 1;
    skip = 1-firstX;

    --initialize outcomes
    xWin = 0;
    oWin = 0;
    tie = 0;

    --initial board states
    p1State = torch.zeros(3,3);
    p2State = torch.zeros(3,3);
    newp1State = torch.zeros(3,3);
    newp2State = torch.zeros(3,3);
    xState = torch.zeros(3,3);
    Q = torch.zeros(3,3);

    --set scores
    winScore = 1;
    loseScore = -1;
    invalidScore = -2;
    tieScore = 0;

    --loop until game ends
    while true do

        if skip==1 then

        end
        

    end




    return score
end

















