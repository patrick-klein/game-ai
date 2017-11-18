
--[[
  class for optimizing hyper-parameters of qLearner

  Class Methods
    __init(net, game)
    train()
    qLearn()
    optimizeNet(batchInputs,batchTargets,actionVals)
    selfEvaluate()
    drawBars(score,divStep,maxScore)
    updateConstants()
    process(input)
    save

]]

-- require libraries
require 'torch'
require 'nn'
require 'optim'
require 'io'

-- require classes
require 'AI/AI'
require 'AI/qLearner'

--globals


-- declare qLearner class
local metaQLearner, parent = torch.class('metaQLearner', 'AI')


-- initialization function
function metaQLearner:__init(game)

  parent.__init(self, game)


  assert(game, 'No game assigned to AI!')

  self.game = game
  self.game.AI = self --turtles all the way down


  numberOfLayers = nil


end



-- public method for training neural network
function metaQLearner:train()
  assert(false, 'metaQLearner:train() method not implemented!')
end

-- method for running test trials
function metaQLearner:selfEvaluate()
  assert(false, 'metaQLearner:selfEvaluate() method not implemented!')
end

--public shorthand for forward pass model
function metaQLearner:process(input)
  assert(false, 'metaQLearner:process() method not implemented!')
end

function metaQLearner:save()
  assert(false, 'metaQLearner:save() method not implemented!')
end

function metaQLearner:returnBestLearner()
  assert(false, 'metaQLearner:returnBestLearner() method not implemented!')
end

--create a new qLearner and assign parameters
function metaQLearner:createQLearner()

  local numHiddenNodes = 1024

  net = nn.Sequential()

  net:add(nn.Linear(self.game.numInputs, numHiddenNodes))
  net:add(nn.ReLU())

  net:add(nn.Linear(numHiddenNodes, numHiddenNodes))
  net:add(nn.ReLU())

  net:add(nn.Linear(numHiddenNodes, self.game.numOutputs))
  --net:add(nn.AddConstant(2))

  --do NOT assign self.game, need to create new instance
  weakLearner = qLearner(self.game.new(), net)

  --weak learner params
  weakLearner.backup = false
  weakLearner.loadMemory = false
  weakLearner.saveMemory = false
  weakLearner.reload = false
  --weakLearner.verbose = false

  weakLearner.numLoopsToFinish = 1000
  weakLearner.numLoopsForLinear = 500
  weakLearner.targetNetworkUpdateDelay = 100
  weakLearner.replayStartSize = 2048
  weakLearner.replaySize = 2048
  weakLearner.batchSize = 2048
  weakLearner.numTrainingEpochs = 5

  weakLearner.eps_initial = 0.9
  weakLearner.eps_final = 0
  weakLearner.gamma_initial = 0.5
  weakLearner.gamma_final = 0.5
  weakLearner.learningRate = 0.00001
  weakLearner:updateConstants()

  return weakLearner

end
