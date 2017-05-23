

--require libraries
require 'torch'
require 'nn'

--require classes
require 'twenty48'
require 'boostLearner'

--set globals
com = 1
hum = 2


--create net
myNet = nn.Sequential()
myNet:add(nn.Linear(16,24))
myNet:add(nn.ReLU())
myNet:add(nn.Linear(24,24))
myNet:add(nn.ReLU())
myNet:add(nn.Linear(24,4))

--create game instance
myGame = twenty48()

--create instance for learner and assign net,game
myAI = boostLearner(myGame)

--train AI
myAI:train()
