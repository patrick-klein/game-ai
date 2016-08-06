-- require libraries
require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
--require 'cudnn'
require 'io'

-- require classes

-- create AI_class
local AI = torch.class('AI')

-- initialization function
function AI:__init(net, game)

	-- required to init
	self.net = net
	self.game = game
	game.AI = self

	-- option to gpu optimize with cuda
	-- possibly infer from net
	--self.cuda = true

	-- display progress
	self.verbose = true
	self.eval = true
	self.profile = true

	-- training parameters
	self.numLoops = 65536
	self.numMem	  = 1024
	self.numMoves = 512
	self.criterion = nn.MSECriterion():cuda()
	self.trainerIteraions = nil

	-- learning constants
	self.eps_initial 		= 0.95		-- eps-greedy value
	local eps_final 		= 0.1
	self.gamma_initial 		= 0.01		-- future reward discount
	local gamma_final 		= 0.5
	self.learnRate_initial 	= 0.01		-- learning rate for gradient descent
	local learnRate_final 	= 0.001
	self.alpha_initial		= 0.001		-- momentum for q-learning (not really momentum is it)
	local alpha_final 		= 0.000001

	self.eps_delta = eps_final-self.eps_initial
	self.gamma_delta = gamma_final-self.gamma_initial
	self.learnRate_delta = learnRate_final-self.learnRate_initial
	self.alpha_delta = alpha_final-self.alpha_initial

	-- memory initialization
	self.memory = {}
	self.memIndex = 0

end


-- private method run once every train loop
function AI:updateConstants()

	-- update learning constants
	self.eps = self.eps_initial+self.eps_delta*(self.loopCount/self.numLoops)
	self.gamma = self.gamma_initial+self.gamma_delta*(self.loopCount/self.numLoops)
	self.learnRate = self.learnRate_initial+self.learnRate_delta*(self.loopCount/self.numLoops)
	self.alpha = self.alpha_initial+self.alpha_delta*(self.loopCount/self.numLoops)

end


function AI:train()

 	-- declare locals
	local flip = false
	local exp
	local rWin local rLose
	local myEval
	local result
	local prevLoopVar = 0

	local playTime
	local trainTime
	local testTime
	local totalTime

	torch.seed()

	sys.tic()

	--loop through episodes (remember one game and train over numMoves moves
	for loopVar = 1,self.numLoops do
		self.loopCount = loopVar

		--update paramaters (or initialize)
		self:updateConstants()

		--wrap dataIndex when full
    	if self.memIndex > self.numMem then
   			flip = true
    		self.replayIndex = 0
    	end
	
		--remember experiences, com-com games
		
		sys.tic()
		self.game:play(com,com)
		playTime = sys.toc()

		--hum,com games
		--[[
		if torch.uniform()>0.5 then
			self.game:play(hum,com)
		else
			self.game:play(com,hum)
		end
		--]]

    	--fill testSet with random values in dataset
		exp = flip and torch.randperm(self.numMem) or torch.randperm(self.memIndex)

		--train net using numMoves number of moves
    	self.replay = {}
		self.numIter = (flip or self.numMoves<self.memIndex) and self.numMoves or self.memIndex			-- set iterations to...
    	for iter=1,self.numIter do														-- if necessary (needed if >5 m...
        	self.replay[iter] = self.memory[ exp[iter] ]
    	end

		--call trainSGD function to update net
		sys.tic()
		self:trainSGD()
		trainTime = sys.toc()

    	--display performance occasionally
    	if true or sys.toc()>10 or loopVar==1 then                              -- displays every 10 seconds

        	--backup twice (in event that save is corrupted from interrupt)
        	torch.save('./saves/myNet.dat', self.net)
        	torch.save('./saves/myNet.dat_bak', self.net)

			if self.verbose then

				if self.eval then
					sys.tic()
					self:test()
					testTime = sys.toc()
				end

				--calculate speed

        		--io.write(loopVar) io.write('\t')
	        	--io.write(sys.toc()/(loopVar-prevLoopVar)) io.write('\n')

				totalTime = playTime + trainTime + testTime
				io.write(loopVar) io.write('\n')
        		io.write('playTime') io.write('\t') io.write(playTime/totalTime) io.write('\n')
        		io.write('trainTime') io.write('\t') io.write(trainTime/totalTime) io.write('\n')
        		io.write('testTime') io.write('\t') io.write(testTime/totalTime) io.write('\n')

				prevLoopVar = loopVar

			end
			sys.tic()

		end
	end
end


function AI:trainSGD()

	local datSize = self.numIter

    local data = {}
    function data:size() return datSize end

	local move
    for move = 1,self.numIter do

        --read values from set
        local origState = self.replay[move][1]
        local nextState = self.replay[move][2]
        local action	= self.replay[move][3]
        local reward	= self.replay[move][4]
        local terminal	= self.replay[move][5]

        --set desired Q value for action
		local y
        if terminal then                            		--terminal gets reward only
            y = reward
        else												--non-terminal adds future (discounted) reward
            local Qnext = self:process(nextState:cuda())	--calculate expected return using current parameters
            y = reward + self.gamma*Qnext:max()
        end

        --calculate Q using current parameters
        local output = self:process(origState:cuda()):double()

        --set target to vector
        local target = output:clone()

        target[action] = (1-self.alpha)*output[action]+self.alpha*y

        data[move] = {origState:cuda(), target:cuda()}

    end

	-- call trainer and train data
    local trainer = nn.StochasticGradient(self.net, self.criterion)
    trainer.learningRate = self.learnRate
	if self.trainerIterations then trainer.maxIterations = self.trainerIterations end
	trainer.verbose = false
    trainer:train(data)

end


function AI:optim() end


function AI:process(input)
	return self.net:forward(input)
end


function AI:test()

	local rWin local rLose
	local myEval

	--evaluate performance
	self.net:evaluate()
	rWin = 0 rLose = 0
	for myEval=1,1e2 do                             --test 100 randomo sample games
		if myEval<50 then                           --half are x
			result = self.game:play(com,challenge)
		else                                        --other half are o
			result = self.game:play(challenge,com)
		end
		if result==win then rWin = rWin+1           --track wins (positive)
		elseif result==lose then rLose = rLose-1    --and losses (negative)
		end
	end
	--io.write(rWin) io.write('\t') io.write(rLose) io.write('\t')
	self.net:training()

end
