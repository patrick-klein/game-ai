
--[[

	This is the parent class for all AI reinforcement learners

	Public Methods
		__init(net,game)
		train()
    selfEvaluate()
		process(input)

]]

-- require libraries
require 'torch'
require 'nn'


-- create AI_class
AI = torch.class('AI')


-- initialization function
function AI:__init(game)

	assert(game, 'No game assigned to AI!')

	self.game = game
	self.game.AI = self				--turtles all the way down

	-- memory initialization
  -- required since all reinforcement learners are based on q-learning
	self.memory = {}
	self.memIndex = 0

end


-- public method for training neural network
function AI:train()
	assert(false,'AI:train() method not implemented!')
end

-- method for running test trials
function AI:selfEvaluate()
	assert(false,'AI:selfEvaluate() method not implemented!')
end

--public shorthand for forward pass model
function AI:process(input)
	assert(false,'AI:process() method not implemented!')
end

function AI:save()
  assert(false,'AI:save() method not implemented!')
end
