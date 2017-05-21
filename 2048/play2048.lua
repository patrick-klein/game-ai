

--require libraries
require 'torch'
--require 'nn'

--require 'cutorch'
--require 'cunn'

--require classes
--require 'AI'
require 'game2048'

com = 1
hum = 2

--create game instance
myGame = game2048()

--create instance for AI_class and assign net
myNet = torch.load('./saves/myNetBest_2048.dat')     --option to load from file
myAI = AI(myNet, myGame)

myGame.draw = true
--torch.manualSeed(123)
myGame:play(com)
