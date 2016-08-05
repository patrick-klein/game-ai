
--dependencies
require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'optim'

--local files
require 'play'
require 'train'
require 'remember'
require 'drawBoard'
require 'challenge'


--create net
if net == nil then

    --load from file
    --net = torch.load('net.dat')

    --build new net
    net = nn.Sequential()
    net:add(nn.Linear(18,1024))
    net:add(nn.Tanh())
    net:add(nn.Linear(1024,512))
    net:add(nn.Tanh())
    net:add(nn.Linear(512,256))
    net:add(nn.Tanh())
    net:add(nn.Linear(256,9))
    net:add(nn.MulConstant(0.01))
    --net:add(nn.HardTanh(-2,2))
    net = net:cuda()

    print("New net created")
    --

    --display net
    print(net)

end

--main script constants
local numLoops = 5e6
local numReplay = 128
local maxIter = 1

--loop initialization
dataSet = {}
dataIndex = 0
io.write('\n')
sys.tic()

--outer loop (loop through 'episodes')
for myLoop = 0,numLoops do

    --learning constants (global)
    eps = 1-0.1*myLoop/numLoops     --eps-greedy value
    gamma = 0.9*myLoop/numLoops     --future reward discount
    learnRate = 0.001-0.000999*myLoop/numLoops

    --wrap dataIndex when full
    if dataIndex > numReplay then
        local flip = true
        dataIndex = 0
    end

    --remember experiences 
    dataSet, dataIndex = play(net, dataSet, dataIndex)

    --fill testSet with random values in dataset
    if flip then
        exp = torch.randperm(numReplay)
        --exp = torch.round(numReplay*torch.uniform())
    else
        exp = torch.randperm(dataIndex)
        --exp = torch.round(dataIndex*torch.uniform())
    end
    testSet = dataSet[exp[1]]

    --train net
    for iter=0,maxIter do
        net = train(net, testSet)
    end

    --backup twice
    torch.save('net.dat', net)
    torch.save('net.dat_bak',net)

    --display running average occasionally
    if myLoop/5e2==torch.round(myLoop/5e2) then
        local win = 0
        local lose = 0
        for myEval=1,1e2 do
            torch.manualSeed(myEval)
            local result = challenge(net)
            if result>0 then
                win = win+1
            elseif result<0 then
                lose = lose-1
            end
        end
        io.write(myLoop) io.write('\t') io.write(win) io.write('\t')
        io.write(lose) io.write('\t') io.write(sys.toc()) io.write('\n')
        sys.tic()
    end
    
end
