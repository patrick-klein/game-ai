

--require libraries
require 'torch'
require 'nn'

--require classes
require 'AI_cpu'
require 'game2048'

--set globals
com = 1
hum = 2

if false then
	myNet = torch.load('./saves/myNetBest_2048.dat')     --option to load from file
else
	-- create Net
	myNet = nn.Sequential()

	--myNet:add(nn.WeightNorm(nn.Linear(16,64)))
	myNet:add(nn.Linear(16,128))
	myNet:add(nn.ReLU())

	--myNet:add(nn.WeightNorm(nn.Linear(256,128)))
	--myNet:add(nn.Linear(64,64))
	--myNet:add(nn.ReLU())

	myNet:add(nn.WeightNorm(nn.Linear(128,64)))
	--myNet:add(nn.Linear(64,64))
	myNet:add(nn.ReLU())

	myNet:add(nn.Linear(64,4))
	--myNet:add(nn.SoftMax())
	--leave last transfer function to game
end

--create game instance
myGame = game2048()

--create instance for AI_class and assign net
myAI = AI(myNet, myGame)

--training parameters


--train for numLoops
--myAI.training = hum
myAI:train()

--myGame:play(com,challenge)
