
--dependencies
require 'torch'
require 'nn'
require 'cunn'
require 'teachTTT'
require 'drawBoard'
--require 'cudatorch';

--create net
if net == nil then
    --net = torch.load('net.dat')
    net = nn.Sequential():cuda()
    net:add(nn.Linear(9,81))
    net:add(nn.ReLU())
    net:add(nn.Linear(81,81))
    net:add(nn.ReLU())
    net:add(nn.Linear(81.9))
    print("New net created")
end

test = torch.Tensor({1, 1, 0, -1, -1, 0, 0, 0, 0})
drawBoard(test)

print(net:forward(test))

repeat

    for loop = 1,2500 do
        net = teachTTT(net, 0.1)
    end

    torch.save('net.dat', net)
    print(net:forward(test))

until false
