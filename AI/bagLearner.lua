--[[

	This is a learner that will use bagging of weak learners

	Class Methods
		__init(net,game)
		train()
    process(input)
    selfEvaluate()
    save

]]

-- require libraries
require 'torch'
require 'nn'

-- require classes
require 'AI/AI'
require 'AI/qLearner'

-- declare boostLearner class
local boostLearner, parent = torch.class('bagLearner', 'AI')


-- initialization function
function bagLearner:__init(game)
  parent.__init(self,game)

	-- number of weak learners to aggregate
  self.numWeakLearners = 3
  self.trainedLearners = 0
  self.learnerPool = {}
end


-- public method for training neural network
function bagLearner:train()

	for t=1,self.numWeakLearners do
    io.write('Training learner #')	io.write(t) io.write('...\n')
		weakLearner = self:createWeakLearner()
    weakLearner:train()
    weakLearner.memory = {}
    weakLearner.memIndex = 0
    weakLearner.game = nil
    self.learnerPool[t] = weakLearner
    self.trainedLearners = self.trainedLearners + 1
    print(self:selfEvaluate())
	end

  --print(self:selfEvaluate())
  self:save()

end


--create a new qLearner and assign parameters
function bagLearner:createWeakLearner()

  net = nn.Sequential()
  net:add(nn.Linear(9,1024))
  net:add(nn.ReLU())
  --net:add(nn.Linear(32,32))
  --net:add(nn.ReLU())
  net:add(nn.Linear(1024,9))

  --do NOT assign self.game, need to create new instance
  weakLearner = qLearner(self.game.new(),net)

  --weak learner params
  weakLearner.backup = false
  weakLearner.loadMemory = false
  weakLearner.saveMemory = false
  --weakLearner.verbose = false
  weakLearner.numLoopsToFinish = 50
  --weakLearner.numLoopsForLinear = 50
  --weakLearner.targetNetworkUpdateDelay = 1e3
  --weakLearner.replayStartSize = 1e4
  --weakLearner.replaySize = 1e5
  --weakLearner.batchSize = 2048

  -- eps-greedy value
  --weakLearner.eps_initial 	= 1
  --weakLearner.eps_final 		= 0.9

  -- future reward discount
  --weakLearner.gamma_initial 	= 0
	--weakLearner.gamma_final 		= 0.05

  weakLearner:updateConstants()

  return weakLearner

end


--public method to return output given a state
function bagLearner:process(input)

  -- average all learner outputs
  vote = torch.Tensor(self.game.numOutputs):zero()
  for t=1,self.trainedLearners do
    vote = vote + self.learnerPool[t]:process(input)/self.trainedLearners
	end
  return vote
end

-- private method for running test trials
function bagLearner:selfEvaluate()

  local numTrials = 1e2
  local runningTotal = 0
  for myEval=1,numTrials do
    runningTotal = runningTotal + self.game:test()
  end

  if self.game.name == '2048' then
    return runningTotal/numTrials

  elseif self.game.name == 'TicTacToe' then
    return (1/2)*(1+runningTotal/numTrials)

  end
end

function bagLearner:save()
  --for t = 1,self.trainedLearners do
  --  self.learnerPool[t].memory = {}
  --  self.learnerPool[t].memIndex = 0
  --  self.learnerPool[t].game = nil
  --end
  torch.save('saves/bagLearner_'..self.game.name..'.ai',self)
end
