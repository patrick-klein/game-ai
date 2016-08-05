
--dependencies
require 'torch'
require 'nn'
require 'cunn'
require 'cutorch'

require 'play'
require 'train'
require 'drawBoard'

--create net
if net == nil then
    --net = torch.load('net.dat')
    net = nn.Sequential()
    net:add(nn.Linear(9,81))
    net:add(nn.Tanh())
    net:add(nn.Linear(81,81))
    net:add(nn.Tanh())
    net:add(nn.Linear(81,9))
    net:add(nn.HardTanh(-2,2))
    print("New net created")
end

--test = torch.Tensor({1, 1, 0, -1, -1, 0, 0, 0, 0})
--drawBoard(test)

--print(net:forward(test))

for myLoop = 1,5 do

    dataset = {}
    dataIndex = 0

    repeat
        dataset, dataIndex = play(net, dataset, dataIndex, 0.5)
    until dataIndex>10e3
    
    
    function dataset:size() return dataIndex end
    net = train(net:cuda(), dataset)
    
    torch.save('net.dat', net)

end
