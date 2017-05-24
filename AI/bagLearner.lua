--[[

	This is a learner that will use bagging of weak learners

	Class Methods
		__init(net,game)
		train()
    process(input)
    selfEvaluate()


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
end


-- public method for training neural network
function bagLearner:train()

	-- number of weak learners to aggregate
	self.numWeakLearners = 10
  self.trainedLearners = 0

  self.learnerPool = {}

	for t=1,self.numWeakLearners do
    io.write('Training learner #')	io.write(t) io.write('...\n')
		weakLearner = self:createWeakLearner()
    weakLearner:train()
    self.learnerPool[t] = weakLearner
    self.trainedLearners = self.trainedLearners + 1
    print(self:selfEvaluate())
	end

  --print(self:selfEvaluate())

end


--create a new qLearner and assign parameters
function bagLearner:createWeakLearner()

  net = nn.Sequential()
  net:add(nn.Linear(16,32))
  net:add(nn.ReLU())
  net:add(nn.Linear(32,32))
  net:add(nn.ReLU())
  net:add(nn.Linear(32,4))

  --do NOT assign self.game, need to create new instance
  weakLearner = qLearner(self.game.new(),net)

  --weak learner params
  weakLearner.backup = false
  weakLearner.verbose = false
  weakLearner.numLoopsToFinish = 2e3
  weakLearner.numLoopsForLinear = 2e3
  weakLearner.targetNetworkUpdateDelay = 5e2
  weakLearner.replayStartSize = 1e4
  weakLearner.replaySize = 1e5
  weakLearner.batchSize = 128

  -- eps-greedy value
  weakLearner.eps_initial 	= 1
  weakLearner.eps_final 		= 0.9
  weakLearner.eps_delta			= weakLearner.eps_final-weakLearner.eps_initial
  weakLearner.eps						= weakLearner.eps_initial

  -- future reward discount
  weakLearner.gamma_initial 	= 0.01
	weakLearner.gamma_final 		= 0.05
	weakLearner.gamma_delta		  = weakLearner.gamma_final-weakLearner.gamma_initial
	weakLearner.gamma 					= weakLearner.gamma_initial

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


-- method for running test trials
function bagLearner:selfEvaluate()

	--intialize locals
	local numTrials = 200
	local runningTotal = 0

	-- random average high score is 67.392, based on 1000 trials
	local randAvg = 67.392

	--find total score across trials
	for myEval=1,numTrials do
		runningTotal = runningTotal+self.game:test()
	end

	--return ratio of AI avg to randAvg
	return (runningTotal/numTrials)/randAvg

end
