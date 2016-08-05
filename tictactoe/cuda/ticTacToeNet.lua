
--torch/lua libraries
require 'torch'
require 'nn'

require 'cutorch'
require 'cunn'

--local files
require 'play'
require 'trainSGD'
require 'challenge'

--create net (unless in torch environment with net predefined)
if net == nil then

    if false then
        net = torch.load('net.dat_bak')     --option to load from file
    else
        net = nn.Sequential()           --or create new net
        net:add(nn.Linear(18,1024))
        net:add(nn.ReLU())
        net:add(nn.Linear(1024,9))
        print("New net created")
    end

    --display net
    print(net)

end

--main script constants
local numLoops = 1024    --number of 'episodes'
local numReplay = 256   --number of moves to remember
local numMoves = 256       --number of moves to train per episode

--loop initialization
local replayMem = {}
local replayIndex = 0

io.write('\n')

sys.tic()

--loop through 'episodes' (remember one game and train over 'numMoves' moves)
for myLoop = 1,numLoops,1 do

    --learning constants (global)
    eps = 1-0.9*myLoop/numLoops             --eps-greedy value                  (1 -> 0.1)
    gamma = 0.01+0.49*myLoop/numLoops       --future reward discount            (0.01 -> 0.5)
    learnRate = 0.01-0.009*myLoop/numLoops  --learnRate for gradient descent    (0.01 -> 0.001)
    alpha = 0.01-0.009*myLoop/numLoops  --momentum for q-learning           (0.001 -> 0.000001)

    --wrap dataIndex when full
    if replayIndex > numReplay then
        flip = true
        replayIndex = 0
    end

    --remember experiences
    torch.seed()                                        --reset seed because manually set elsewhere
    replayMem, replayIndex = play(net, replayMem, replayIndex)

    --fill testSet with random values in dataset
    local exp = flip and torch.randperm(numReplay) or torch.randperm(replayIndex)

    --train net using 'numMoves' number of moves
    testSet = {}
    numIter = (flip or numMoves<replayIndex) and numMoves or replayIndex  --set iterations to 'replayIndex'
    for iter=1,numIter do                                                 --if necessary (needed if >5 moves)
        testSet[iter] = replayMem[exp[iter]]
    end

    net = trainSGD(net, testSet, numIter)

    --display performance occasionally
    --if myLoop/5e2==torch.round(myLoop/5e2) then     --currently configured to display every 500 iterations
    if sys.toc()>10 then                              --displays every 10 seconds

        --net:evaluate()
        --local dWin = 0 local dLose = 0
        --local rWin = 0 local rLose = 0
        --for myEval=1,1e2 do                         --test 100 sample games (deterministic across samplings)

            --deterministic trial
            --torch.manualSeed(myEval)
            --local result = challenge(net)
            --if result>0 then dWin = dWin+1            --track wins (positive)
            --elseif result<0 then dLose = dLose-1      --and losses (negative)
            --end

            --random trial
            --torch.seed()
            --local result = challenge(net)
            --if result>0 then rWin = rWin+1            --track wins (positive)
            --elseif result<0 then rLose = rLose-1      --and losses (negative)
            --end

        --end

        --myLoop    dWin    dLose   rWin    rLose   toc
        io.write(myLoop) io.write('\t') 
        --io.write(dWin) io.write('\t') io.write(dLose) io.write('\t')
        --io.write(rWin) io.write('\t') io.write(rLose) io.write('\t')
        io.write(sys.toc()) io.write('\n')
	sys.tic()

        --backup twice (in event that save is corrupted from interrupt)
        net:training()
        torch.save('net.dat', net)
        torch.save('net.dat_bak',net)

    end
    

end
