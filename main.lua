--[[
  Creates an instance of an AI learner, then trains it
  Saves and plays a game when finished

  To-Do / Considerations
    Use two separate AI modules for 2-player games
    Implement CUDA
    create method that scrapes out unnecessary data when saving
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
----should move this into qLearner
if false then
  myNet = torch.load('./saves/myNetBest_2048.dat_archive')
  print(myNet)
else
  myNet = nn.Sequential()
  myNet:add(nn.Linear(16, 1024))
  myNet:add(nn.ReLU())
  myNet:add(nn.Linear(1024, 1024))
  myNet:add(nn.ReLU())
  myNet:add(nn.Linear(1024, 4))
end


--create game instance
myGame = twenty48()
--myGame = ticTacToe()


--create instance for AI and assign net,game
--myAI = qLearner(myGame, myNet)
--myAI.loadMemory = true
myAI = bagLearner(myGame)

--train learner
myAI:train()

--should create different output/save folders for each game

--save ai and play game
myAI:save()
myGame.draw = true
myGame:play(com, com)
