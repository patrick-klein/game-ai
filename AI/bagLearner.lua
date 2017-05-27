
--[[
  This is a learner that will use bagging of weak learners

  Class Methods
    __init(net, game)
    train()
    process(input)
    selfEvaluate()
    learnerEvaluate()
    save()

]]

-- require libraries
require 'torch'
require 'nn'

-- require classes
require 'AI/AI'
require 'AI/qLearner'

-- declare boostLearner class
local bagLearner, parent = torch.class('bagLearner', 'AI')


-- initialization function
function bagLearner:__init(game)
  parent.__init(self, game)

  --number of weak learners to aggregate
  self.numWeakLearners = 10

  --variable initializations
  self.trainedLearners = 0
  self.learnerPool = {}
  self.learnerWeights = torch.Tensor(self.numWeakLearners):zero()

end


--method for training learners in AI module
function bagLearner:train()

  --create and train learners
  for t = 1, self.numWeakLearners do
    io.write('Training learner #') io.write(t) io.write('...\n')
    weakLearner = self:createWeakLearner()
    weakLearner:train()

    --need to free up memory, esp. if using many learners
    weakLearner.memory = nil
    weakLearner.memIndex = nil
    collectgarbage()

    --add learner to pool
    self.learnerPool[t] = weakLearner
    self.trainedLearners = self.trainedLearners + 1

    --get a metric of this learners performance, weight accordingly
    self:learnerEvaluate(t)
    print(self.learnerWeights)
    print(self:selfEvaluate())
  end

end


--create a new qLearner and assign parameters
function bagLearner:createWeakLearner()

  net = nn.Sequential()
  net:add(nn.Linear(self.game.numInputs, 1024))
  net:add(nn.PReLU(1024))
  --net:add(nn.ReLU())
  --net:add(nn.Linear(256,256))
  --net:add(nn.ReLU())
  net:add(nn.Linear(1024, self.game.numOutputs))

  --do NOT assign self.game, need to create new instance
  weakLearner = qLearner(self.game.new(), net)

  --weak learner params
  weakLearner.backup = false
  weakLearner.loadMemory = false
  weakLearner.saveMemory = false
  --weakLearner.verbose = false

  weakLearner.numLoopsToFinish = 150
  weakLearner.numLoopsForLinear = 150
  weakLearner.targetNetworkUpdateDelay = 75
  --weakLearner.replayStartSize = 1e4
  --weakLearner.replaySize = 1e5
  --weakLearner.batchSize = 2048

  weakLearner.eps_initial = 1
  weakLearner.eps_final = 0.5
  weakLearner.gamma_initial = 0
  weakLearner.gamma_final = 0.25
  weakLearner:updateConstants()

  return weakLearner

end


--public method to return output given a state
function bagLearner:process(input)

  -- average all learner outputs
  vote = torch.Tensor(self.game.numOutputs):zero()
  for t = 1, self.trainedLearners do
    vote = vote + self.learnerWeights[t] * self.learnerPool[t]:process(input)
  end
  return vote
end

-- private method for running test trials
function bagLearner:selfEvaluate()

  local numTrials = 1e2
  local runningTotal = 0
  for myEval = 1, numTrials do
    torch.manualSeed(42 * myEval)
    runningTotal = runningTotal + self.game:test()
  end
  torch.seed()

  if self.game.name == '2048' then
    return runningTotal / numTrials

  elseif self.game.name == 'TicTacToe' then
    return (1 / 2) * (1 + runningTotal / numTrials)

  end
end


function bagLearner:learnerEvaluate()

  ----for increased efficency, only evaluate one learner
  ----  need a way to re-weight accordingly
  ----  maybe rescale scores to ensure using softlogmax

  for t = 1, self.trainedLearners do
    local numTrials = 1e2
    local runningTotal = 0
    for myEval = 1, numTrials do
      torch.manualSeed(42 * myEval)
      runningTotal = runningTotal + self.learnerPool[t].game:test()
    end
    if self.game.name == '2048' then
      score = runningTotal / numTrials
    elseif self.game.name == 'TicTacToe' then
      score = (1 / 2) * (1 + runningTotal / numTrials)
    end
    self.learnerWeights[t] = score
  end
  torch.seed()

  self.learnerWeights = self.learnerWeights / self.learnerWeights:sum()

end

function bagLearner:save()
  for t = 1, self.trainedLearners do
    self.learnerPool[t].game = nil
  end
  torch.save('saves/bagLearner_'..self.game.name..'.ai', self)
end
