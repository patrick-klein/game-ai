--[[
	Creates an instance of an AI learner, then trains it
	Saves and plays a game when finished
]]


--require libraries
require 'torch'
require 'nn'

--require classes
require 'AI/qLearner'
require 'AI/bagLearner'
require 'games/twenty48'
require 'games/ticTacToe'


--set globals
com = 1
hum = 2


--quickly change between new and archived networks
if false then
	myNet = torch.load('./saves/myNetBest_2048.dat_archive')
	print(myNet)
else
	myNet = nn.Sequential()
	myNet:add(nn.Linear(9,128))
	myNet:add(nn.ReLU())
	--myNet:add(nn.Linear(256,256))
	--myNet:add(nn.ReLU())
	myNet:add(nn.Linear(128,9))
end


--create game instance
--myGame = twenty48()
myGame = ticTacToe()


--create instance for AI and assign net,game
myAI = qLearner(myGame, myNet)
--myAI = bagLearner(myGame)

--train learner
myAI:train()

--save ai and play game
torch.save('./saves/qLearner_ticTacToe.ai', myAI)
myGame.draw = true
myGame:play(com)