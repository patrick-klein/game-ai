
--[[

	Inherited Methods
		process(input)

	Private Methods
		__init(net, game)
		train()
		qLearn()
		optim(batchInputs,batchTargets,actionVals)
		selfEvaluate()
		updateConstants()

]]

-- require libraries
require 'torch'
require 'nn'
require 'optim'
require 'io'

-- require classes
require 'AI'


--globals


-- declare qLearner class
local qLearner, parent = torch.class('qLearner', 'AI')


-- initialization function
function qLearner:__init(game, net)

	parent.__init(self, game)

	assert(net, 'No neural network to initialize AI!')
	self.net = net
	self.targetNet = self.net:clone()

	--use AI generated moves for training, by default
	self.training = com

	-- flags for
	self.profile = false
	self.backup = true
	self.verbose = true

	-- training parameters (look up Deep Mind paper for explanations)
	self.numLoopsToFinish = 1e6
	self.numLoopsForLinear = 1e5
	self.firstLoop = 1
	self.targetNetworkUpdateDelay = 1e3
	self.replayStartSize = 5e3
	self.replaySize = 1e5
	self.batchSize = 128

	-- eps-greedy value
	self.eps_initial 		= 1.0
	local eps_final 		= 0.1
	self.eps_delta			= eps_final-self.eps_initial
	self.eps						= self.eps_initial

	-- future reward discount
	self.gamma_initial 	= 0.01
	local gamma_final 	= 0.50
	self.gamma_delta		= gamma_final-self.gamma_initial
	self.gamma 					= self.gamma_initial

end

-- public method for training neural network
function qLearner:train()

 	--declare locals
	local wrapMemory = false
	local playTime	= 0
	local learnTime	= 0
	local testTime	= 0
	local totalTime	= 0
	local accTime	= 0
	local score			= 0
	local bestScore		= 0
	local tallyScore	= 0
	local tallyCount	= 0
	local avgScore		= 0
	local prevAvgScore	= 0
	local avgQ			= 0
	local avgQCount		= 0
	self.maxQ = 0

	--new random seed
	torch.seed()

	--loop through episodes (playing games to increase memory, then train over memories)
	for loopVar = self.firstLoop,self.numLoopsToFinish do
		self.loopCount = loopVar

		--update paramaters
		if loopVar < self.numLoopsForLinear then
			self:updateConstants()
		end

		--wrap dataIndex when full
  	if self.memIndex > self.replaySize then
 			wrapMemory = true
  		self.memIndex = 0
  	end

		--option to load memory dump (contains intialized memories from prior calls)
		--used during testing since it can take a while to generate
		if false and loopVar==1 then
			self.memory = torch.load('./saves/memoryDump.dat')
			self.memIndex = self.replayStartSize
		end

		--play game until memory is initialized, or batchSize reached
		sys.tic()
		self.net:evaluate()
		self.game.testmode = false
		if wrapMemory or self.memIndex>=self.replayStartSize then
			fillMemory = self.memIndex+self.batchSize
		else
			if self.verbose then
				io.write('\n')
				print('Initializing replay memory, this may take some time...')
			end
			fillMemory = self.replayStartSize
		end
		repeat self.game:play(com)
		until self.memIndex>=fillMemory
		playTime = sys.toc()

		--option to save memory dump
		if true and loopVar==1 then
			torch.save('./saves/memoryDump.dat', self.memory)
		end

		--fill testSet with random values in dataset
		self.replay = {}
		randMems = wrapMemory and torch.randperm(self.replaySize) or torch.randperm(self.memIndex)
   	for i=1,self.batchSize do
			self.replay[i] = self.memory[randMems[i]]
		end

		--reset targetNet to enforce off-policy learning (look up Deep Mind paper)
		if loopVar%self.targetNetworkUpdateDelay==0 then
			self.targetNet = self.net:clone()
			if self.verbose then
				io.write('\n')
				print('Updating target network.')
			end
		end

		--episode of q-learning
		sys.tic()
		avgQ = avgQ + self:qLearn()
		avgQCount = avgQCount + self.batchSize
		learnTime = sys.toc()

		--record time
		accTime = accTime+playTime+learnTime

    --display performance occasionally
    if (self.verbose or self.backup) and (accTime>20 or loopVar==self.firstLoop) then

      --backup twice (in event that save is corrupted from interrupt)
			torch.save('./saves/myNet_2048.dat', self.net)
      torch.save('./saves/myNet_2048.dat_bak', self.net)

			--test performance against trials
			sys.tic()
			score = self:selfEvaluate()
			testTime = sys.toc()

			--save another copy if new high score
			if score>=bestScore then
				torch.save('./saves/myNetBest_2048.dat', self.net)
				torch.save('./saves/myNetBest_2048.dat_bak', self.net)
				bestScore = score
			else
				--reload best model if performance declined
				--self.net = torch.load('./saves/myNetBest_2048.dat')
			end

			--keep running average
			tallyScore = tallyScore + score
			tallyCount = tallyCount + 1
			avgScore = tallyScore/tallyCount

			--display iteration, score, and running average
			--annotate with * for new bestScore, or - for declining average
			io.write('\n')
			io.write('\tIteration\t')	io.write(loopVar)			io.write('\n')
			if score==bestScore then io.write('*') end
			io.write('\tCurrent Score\t')	io.write(score)				io.write('\n')
			if avgScore<prevAvgScore then io.write('-') end
			io.write('\tAverage Score\t')	io.write(avgScore)			io.write('\n')
			io.write('\tAverage Q\t')	io.write(avgQ/avgQCount)	io.write('\n')

			--printing maxQ
			--might be useful for testing
			--print(self.maxQ)
			self.maxQ = 0

			--update averages
			prevAvgScore = avgScore
			avgQ = 0
			avgQCount = 0

			--display code execution times if profiling
			if self.profile then
				totalTime = playTime + learnTime + testTime
				io.write('\n')
   			io.write('playTime') io.write('\t') io.write(playTime/totalTime)	io.write('\n')
   			io.write('trainTime') io.write('\t') io.write(learnTime/totalTime)	io.write('\n')
   			io.write('testTime') io.write('\t') io.write(testTime/totalTime)	io.write('\n')
			end

			--reset timer
			accTime = 0
		end
	end
end



--private method to recall memories and set target, calls optimization function
function qLearner:qLearn()

	--create tensors to hold inputs and targets for optimization function
	local batchInputs = torch.Tensor(self.batchSize,self.game.numInputs)
	local batchTargets = torch.Tensor(self.batchSize,self.game.numOutputs)

	--initialize average Q value
	local avgQ = 0

	--must be set to evaluate, need to get future Q values
	self.net:evaluate()
	self.targetNet:evaluate()
	self.game.testmode = true
    for move = 1,self.batchSize do

        --read values from set
        local origState = self.replay[move][1]			--passed directly into batchInputs
        local nextState = self.replay[move][2]
        local action	= self.replay[move][3]
        local reward	= self.replay[move][4]
        local terminal	= self.replay[move][5]

        --set desired Q value for action
		local y
		local Qnext
        if terminal then
			--terminal gets reward only
            y = reward
        else
			--non-terminal adds future discounted reward,
			Qnext = self.targetNet:forward(nextState)	--calculate expected return using targetNet
            y = reward + self.gamma*Qnext:max()
        end

		--update average Q
		avgQ = avgQ + y
		if y > self.maxQ then self.maxQ = y end

		----debug: checking for nil value
		if avgQ~=avgQ then
			print(Qnext)
			print(self.targetNet)
			print(avgQ)
			print(y)
			print(self.gamma)
			print(origState)
			print(nextState)
			print(action)
			print(reward)
			print(terminal)
			assert(false)
		end

		--calculate Q using current parameters
		local output = self:process(origState)

		--set target to current Q
		local target = output:clone()

		--adjust value of current action
		--clamp delta to +1/-1
		if torch.abs(y-output[action])<1 or terminal then
			target[action] = y
		else
			----is there a better way to check sign(y-output[action])?
			target[action] = output[action]+(y-output[action])/torch.abs(y-output[action])
		end

		--DEBUG--
		--target = torch.zeros(4)
		--target[action] = y

		--package input and targets for batch optimization
		batchInputs[move] = origState
		batchTargets[move] = target

    end

	--DEBUG--
	--print(torch.round(1000*qVal/self.batchSize)/1000)

	--batch optimization function
	self:optim(batchInputs,batchTargets)

	--return average Q value
	return avgQ

end


--private method for batch back propagation and parameter optimization, only one iteration
function qLearner:optim(batchInputs,batchTargets,actionVals)

	--set net to training
	self.net:training()

	--set training settings
	--local config = {}
	local config = {learningRate = 0.00025}
	local criterion = nn.MSECriterion()
	local params,gradParams = self.net:getParameters()

	--create function that returns loss,gradParams for optimization
	local function feval(params)
		gradParams:zero()
		local outputs = self.net:forward(batchInputs)
		if outputs:ne(outputs):sum()>0 then
			print(outputs)
			assert(false)
		end
		local loss = criterion:forward(outputs, batchTargets)
		local dloss_doutput = criterion:backward(outputs,batchTargets)
		--need to find way to restrict dloss_doutput to one output
		--dloss_doutput[1-
		self.net:backward(batchInputs,dloss_doutput)
		return loss,gradParams
	end

	--apply optimization method
	--optim.sgd(feval,params,config)
	--optim.adamax(feval,params,config)
	optim.rmsprop(feval,params,config)

end

-- private method for running test trials
function qLearner:selfEvaluate()

	--intialize locals
	local numTrials = 20
	local runningTotal = 0

	-- random average high score is 67.392, based on 1000 trials
	local randAvg = 67.392

	--set net to evaluate, and game to testmode
	self.net:evaluate()
	self.game.testmode = true

	--find total score across trials
	--self.game.draw = true
	for myEval=1,numTrials do
		--torch.manualSeed(myEval*123)
		runningTotal = runningTotal+self.game:test()
	end
	self.game.draw = false

	--new random seed
	torch.seed()

	--return ratio of AI avg to randAvg
	return (runningTotal/numTrials)/randAvg

end


-- private method run once every train loop
function qLearner:updateConstants()

	-- update learning constants
	self.eps = self.eps_initial+self.eps_delta*(self.loopCount/self.numLoopsForLinear)
	self.gamma = self.gamma_initial+self.gamma_delta*(self.loopCount/self.numLoopsForLinear)

end
