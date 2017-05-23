--[[

	This is the parent class for all AI reinforcement learners

	Parent Methods
		process(input)

	class Methods
		__init(net,game)
		train()

]]

-- require libraries
require 'torch'
require 'nn'

-- require classes
require 'AI'

-- declare boostLearner class
local boostLearner, parent = torch.class('boostLearner', 'AI')


-- initialization function
function boostLearner:__init(game)

	parent.__init(self,game)

end


-- public method for training neural network
function boostLearner:train()
	assert(false,'boostLearner:train() method not implemented!')
end
