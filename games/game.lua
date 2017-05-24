--[[
	Parent class for all games

	Public Methods
		play(player)
		test()

]]


-- require libraries
require 'torch'
require 'nn'
require 'io'

-- create game
local game = torch.class('game')


-- initialization function
function game:__init()

	self.AI = nil
	self.testmode = false

  self.name = nil

	self.numInputs   = nil
	self.numOutputs  = nil
  self.numPlayers = nil

  self.draw = nil

  self.maxScore = nil

end


--public method that plays a game, player can be human or AI
--must return a score
function game:play()
	assert(false, 'game:play(player) method not implemented!')
end


--public method for running trials on AI
--must return a score
function game:test()
	assert(false, 'game:test() method not implemented!')
end
