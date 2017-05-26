
--[[
  Script that is used for one-off testing purposes

]]

--require libraries
require 'torch'
require 'nn'

--require classes
require 'games/twenty48'
require 'games/ticTacToe'
require 'AI/bagLearner'
require 'AI/qLearner'

--set globals
com = 1
hum = 2


--create net
myNet = nn.Sequential()
myNet:add(nn.Linear(9, 32))
myNet:add(nn.ReLU())
--myNet:add(nn.Linear(24,24))
--myNet:add(nn.ReLU())
myNet:add(nn.Linear(32, 9))

--create game instance
--myGame = twenty48()
myGame = ticTacToe()


--create instance for learner and assign net,game
--myAI = bagLearner(myGame)
myAI = qLearner(myGame, myNet)
myAI.

--train AI
myAI:train()
