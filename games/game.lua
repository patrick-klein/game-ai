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


-- require classes


-- create game
local game = torch.class('game')


-- initialization function
function game:__init()

	self.AI = nil
	self.testmode = false

end


--public method that plays a game, player can be human or AI
--must return a score
function game:play(player)

	assert(false, 'game:play(player) method not implemented!')

end


--public method for running trials on AI
function game:test()
	self.testmode = true
	score = self:play(com)
	self.testmode = false
	return score
end
