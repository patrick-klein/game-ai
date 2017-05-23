

--[[
	This script creates an instance of the game
	Using the if statement, either a human or AI will play the game
]]

--require libraries
require 'torch'

--require classes
require 'qLearner'
require 'twenty48'

--globals
com = 1
hum = 2

--create game instance
myGame = twenty48()

--quick way to choose human or AI player
if false then
	myGame:play(hum)
else
	myNet = torch.load('./saves/myNetBest_2048.dat_archive')     --option to load from file
	myAI = qLearner(myGame, myNet)
	myGame.draw = true
	myGame:play(com)
end
