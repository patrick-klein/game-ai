--[[
	Creates an instance of an AI learner, then trains it
	Play a game when finished
]]


--require libraries
require 'torch'
require 'nn'

--require classes
require 'qLearner'
require 'bagLearner'
require 'twenty48'


--set globals
com = 1
hum = 2


--quickly change between new and archived networks
if false then
	myNet = torch.load('./saves/myNetBest_2048.dat_archive')
else
	-- create Net
	myNet = nn.Sequential()
	myNet:add(nn.Linear(16,32))
	myNet:add(nn.ReLU())
	myNet:add(nn.Linear(32,32))
	myNet:add(nn.ReLU())
	--myNet:add(nn.WeightNorm(nn.Linear(128,64)))
	--myNet:add(nn.ReLU())
	myNet:add(nn.Linear(32,4))
	--myNet:add(nn.SoftMax())
	--leave last transfer function to game
end


--create game instance
myGame = twenty48()


--create instance for AI and assign net,game
--myAI = qLearner(myGame, myNet)
myAI = bagLearner(myGame)

--training parameters
--myAI.training = hum

--train learner
myAI:train()

--play game
torch.save('./saves/bagLearner.ai', myAI)
myGame.draw = true
myGame:play(com)