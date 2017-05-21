
-- require libraries
require 'torch'
require 'nn'
require 'optim'
require 'io'

-- require classes

--globals

-- create AI_class
local AI = torch.class('AI')


-- initialization function
function AI:__init(net, game)

	-- required to init
	self.net = net
	self.targetNet = self.net:clone()
	self.game = game
	self.game.AI = self

	-- option to gpu optimize with cuda
	-- possibly infer from net
	--self.cuda = true

	self.training = com

	-- display progress
	self.profile = false

	-- training parameters
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
	self.eps 				= self.eps_initial

	-- future reward discount
	self.gamma_initial 		= 0.01
	local gamma_final 		= 0.99
	self.gamma_delta		= gamma_final-self.gamma_initial
	self.gamma 				= self.gamma_initial

	-- memory initialization
	self.memory = {}
	self.memIndex = 0

end

-- public method for training neural network
function AI:train()

 	-- declare locals
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


	torch.seed()

	--loop through episodes (play one game and train one batch)
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

		--option to load memory dump
		if false and loopVar==1 then
			self.memory = torch.load('./saves/memoryDump.dat')
			self.memIndex = self.replayStartSize
		end

		--episode of play
		sys.tic()
		self.net:evaluate()
		self.game.testmode = false
		if wrapMemory or self.memIndex>=self.replayStartSize then
			fillMemory = self.memIndex+self.batchSize
		else
			io.write('\n')
			print('Initializing replay memory, this may take some time...')
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

		--episode of q-learning
		if loopVar%self.targetNetworkUpdateDelay==0 then
			self.targetNet = self.net:clone()
			io.write('\n')
			print('Updating target network.')
		end
		sys.tic()
		avgQ = avgQ + self:qLearn()
		avgQCount = avgQCount + self.batchSize
		learnTime = sys.toc()

		--record time
		accTime = accTime+playTime+learnTime

    	--display performance occasionally
    	if accTime>20 or loopVar==self.firstLoop then		-- displays every x seconds

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
			io.write('\n')
			io.write('\t')	io.write(loopVar)			io.write('\n')
			if score==bestScore then io.write('*') end
			io.write('\t')	io.write(score)				io.write('\n')
			if avgScore<prevAvgScore then io.write('-') end
			io.write('\t')	io.write(avgScore)			io.write('\n')
			io.write('\t')	io.write(avgQ/avgQCount)	io.write('\n')

			print(self.maxQ)
			self.maxQ = 0

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

			accTime = 0
		end
	end
end



--private method to recall memories and set target.  calls optimization function
function AI:qLearn()

	local batchInputs = torch.Tensor(self.batchSize,self.game.numInputs)
	local batchTargets = torch.Tensor(self.batchSize,self.game.numOutputs)

	local avgQ = 0

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
			--non-terminal adds future (discounted) reward
			Qnext = self.targetNet:forward(nextState)	--calculate expected return using current parameters
            y = reward + self.gamma*Qnext:max()
        end
		avgQ = avgQ + y
		if y > self.maxQ then self.maxQ = y end

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
			target[action] = output[action]+(y-output[action])/torch.abs(y-output[action])
		end

		--DEBUG--
		--target = torch.zeros(4)
		--target[action] = y

		--package input and targets for batch optimization
		batchInputs[move] = origState
		batchTargets[move] = target

    end

	--print(torch.round(1000*qVal/self.batchSize)/1000)

	--batch optimization function
	self:optim(batchInputs,batchTargets)

	return avgQ

end


--private method for batch back propagation and parameter optimization
function AI:optim(batchInputs,batchTargets,actionVals)


	self.net:training()

	--local config = {}
	local config = {learningRate = 0.00025}
	local criterion = nn.MSECriterion()
	local params,gradParams = self.net:getParameters()

	local function feval(params)
		gradParams:zero()
		local outputs = self.net:forward(batchInputs)
		if outputs:ne(outputs):sum()>0 then
			print(outputs)
			assert(false)
		end
		local loss = criterion:forward(outputs, batchTargets)
		local dloss_doutput = criterion:backward(outputs,batchTargets)
		--dloss_doutput[1-
		self.net:backward(batchInputs,dloss_doutput)
		return loss,gradParams
	end

	--optim.sgd(feval,params,config)
	--optim.adamax(feval,params,config)
	optim.rmsprop(feval,params,config)

end

-- private method for running test trials
function AI:selfEvaluate()

	local numTrials = 20
	local runningTotal = 0

	-- random average high score is 67.392, based on 1000 trials
	local randAvg = 67.392

	self.net:evaluate()
	self.game.testmode = true

	--self.game.draw = true
	for myEval=1,numTrials do
		--torch.manualSeed(myEval*123)
		runningTotal = runningTotal+self.game:test()
	end
	self.game.draw = false

	torch.seed()
	return (runningTotal/numTrials)/randAvg

end


-- private method run once every train loop
function AI:updateConstants()

	-- update learning constants
	self.eps = self.eps_initial+self.eps_delta*(self.loopCount/self.numLoopsForLinear)
	self.gamma = self.gamma_initial+self.gamma_delta*(self.loopCount/self.numLoopsForLinear)

end


--public shorthand for forward pass model
function AI:process(input)
	return self.net:forward(input)
end


