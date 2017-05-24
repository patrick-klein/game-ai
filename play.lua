

--[[
	This script creates an instance of the game
	Using the if statement, either a human or AI will play the game
]]

--require libraries
require 'torch'

--require classes
require 'AI/qLearner'
require 'games/twenty48'
require 'games/ticTacToe'

--globals
com = 1
hum = 2

--create game instance
--myGame = twenty48()
myGame = ticTacToe()

myNet = nn.Sequential()
myNet:add(nn.Linear(9,32))
myNet:add(nn.ReLU())
myNet:add(nn.Linear(32,9))
myAI = qLearner(myGame, myNet)

--quick way to choose human or AI player
if false then
	myGame:play(hum,com)
else
	myGame.draw = true
	myGame:play(com,com)
end
