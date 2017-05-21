

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

if true then
	myNetCuda = torch.load('./saves/myNetBest.dat')     --option to load from file
else
	-- create Net
	myNet = nn.Sequential()

	myNet:add(nn.Linear(9,1024))
	myNet:add(nn.ReLU())
	--myNet:add(nn.Dropout(0.5))

	--myNet:add(nn.Linear(2048,1024))
	--myNet:add(nn.ReLU())
	--myNet:add(nn.Dropout(0.4))

	myNet:add(nn.Linear(1024,9))
	--myNet:add(nn.Tanh())

	--leave last transfer function to game
	myNetCuda = myNet:cuda()
end

--display net
--print(myNetCuda)

--create game instance
myGame = ticTacToe()

--create instance for AI_class and assign net
myAI = AI(myNetCuda, myGame)

--training parameters


--train for numLoops
myAI.training = hum
myAI:train()

--myGame:play(com,challenge)
