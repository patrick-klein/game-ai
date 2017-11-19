--[[
  Creates an instance of an AI learner, then trains it
  Saves and plays a game when finished

  To-Do / Considerations
    Use two separate AI modules for 2-player games
    Implement CUDA
    create method that scrapes out unnecessary data when saving
    pass class of game instead of instance to AI agents
    use actual high score for 2048 game
    augment tic tac toe memories
    update bagLearner to use config
]]


--require libraries
require 'torch'
require 'nn'

--require classes
require 'AI/qLearner'
require 'AI/bagLearner'
require 'AI/metaQLearner'
require 'games/twenty48'
require 'games/ticTacToe'


--set globals
com = 1
hum = 2


--option to load pretrained network
if false then
  myNet = torch.load('./saves/myNetBest_2048.dat_archive')
  print(myNet)
end


--create game instance
--myGame = twenty48()
myGame = ticTacToe()


--create instance for AI and assign net,game
--myAI = qLearner(myGame, myNet)
--myAI.loadMemory = true
--myAI = bagLearner(myGame)

params = {numHiddenNodes={256, 512, 1024},
          numberOfLayers={1,2,3},
          numLoopsToFinish={50},
          }

myAI = metaQLearner(myGame, params)

--train learner
myAI:train()

--save ai and play game
myAI:save()
myGame.draw = true
myGame:play(com, com)
