

--[[
  This script creates an instance of the game
  Using the if statement, either a human or AI will play the game
  
]]

--require libraries
require 'torch'

--require classes
require 'AI/qLearner'
require 'AI/bagLearner'
require 'games/twenty48'
require 'games/ticTacToe'

--globals
com = 1
hum = 2

--create game instance
myGame = twenty48()
--myGame = ticTacToe()

myAI = torch.load('saves/bagLearner_2048.ai')
myGame.AI = myAI

--quick way to choose human or AI player
if false then
  myGame:play(com, hum)
else
  myGame.draw = true
  myGame:play(com)
end
