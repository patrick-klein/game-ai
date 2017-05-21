

--require libraries
require 'torch'
require 'nn'

require 'cutorch'
require 'cunn'

--require classes
require 'AI'
require 'ticTacToe'

--set globals
win  =  1
lose = -1
draw =  0

com = 1
hum = 2
challenge = 3

myNetCuda = torch.load('./saves/myNetBest.dat_archive1')     --option to load from file

--create game instance
myGame = ticTacToe()

--create instance for AI_class and assign net
myAI = AI(myNetCuda, myGame)

--training parameters
myAI.eps = 0

myGame:play(hum,com)
myGame:play(com,hum)
