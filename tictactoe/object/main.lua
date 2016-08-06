

-- require libraries
require 'torch'
require 'nn'

require 'cutorch'
require 'cunn'

-- require classes
require 'AI'
require 'ticTacToe'

-- set globals
win  =  1
lose = -1
draw =  0

com = 1
hum = 2
challenge = 3

-- create Net
myNet = nn.Sequential()
myNet:add(nn.Linear(18,1024))
myNet:add(nn.ReLU())
--myNet:add(nn.Dropout(0.2))
myNet:add(nn.Linear(1024,1024))
myNet:add(nn.ReLU())
--myNet:add(nn.Dropout(0.5))
--myNet:add(nn.Linear(128,128))
--myNet:add(nn.ReLU())
--myNet:add(nn.Dropout(0.2))
myNet:add(nn.Linear(1024,9))

myNetCuda = myNet:cuda()

-- create game instance
myGame = ticTacToe()

-- create instance for AI_class and assign net
myAI = AI(myNetCuda, myGame)

-- training parameters
myAI.numLoops = 65536
myAI.numMem   = 512
myAI.numMoves = 128
myAI.criterion = nn.MSECriterion():cuda()

-- train for numLoops
myAI:train()

--myGame:play(com,challenge)
