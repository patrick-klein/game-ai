

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
myNet:add(nn.Linear(18,2048))
myNet:add(nn.ReLU())
myNet:add(nn.Linear(2048,1024))
myNet:add(nn.ReLU())
myNet:add(nn.Linear(1024,9))

myNetCuda = myNet:cuda()

-- create game instance
myGame = ticTacToe()

-- create instance for AI_class and assign net
myAI = AI(myNetCuda, myGame)

-- set constants for AI_class:train
-- leave defaults for now

-- train until some end condition is met
myAI:train()
