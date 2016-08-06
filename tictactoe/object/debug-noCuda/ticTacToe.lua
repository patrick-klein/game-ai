-- require libraries
require 'torch'
require 'nn'

require 'io'

-- require classes
require 'AI'

-- create ticTacToe_class
local ticTacToe = torch.class('ticTacToe')

-- local constants
xTurn = 1
yTurn = 2

-- victory types
victory = torch.Tensor(8,3);
victory[1] = torch.Tensor({1, 2, 3})
victory[2] = torch.Tensor({4, 5, 6})
victory[3] = torch.Tensor({7, 8, 9})
victory[4] = torch.Tensor({1, 4, 7})
victory[5] = torch.Tensor({2, 5, 8})
victory[6] = torch.Tensor({3, 6, 9})
victory[7] = torch.Tensor({1, 5, 9})
victory[8] = torch.Tensor({3, 5, 7})


-- initialization function
function ticTacToe:__init()
	
    -- set scores
    self.winScore = 1
    self.loseScore = -1
    self.invalidScore = -1
    self.tieScore = 0
    self.noScore = 0

	self.xWin = false
	self.oWin = false
	self.tie  = false

end


function ticTacToe:play(p1,p2,AI)

	self.p1 = p1
	self.p2 = p2

	assert(self.p1, 'Player 1 must be set')
	assert(self.p2, 'Player 2 must be set')

    --set turn variable
    self.turn = 1

	if self.p1 == com and self.p2 == com then
		assert(AI, 'CvC needs AI')
		return self:play_CvC(AI)

	elseif self.p1 == hum and self.p2 == hum then
		return self:play_PvP()

	else
		assert(AI, 'PvC needs AI')
		return self:play_PvC()

	end
end



function ticTacToe:play_CvC(AI)


    --initial board states
    self.p1State = torch.zeros(18)
    self.p2State = torch.zeros(18)
    self.newp1State = torch.zeros(18)
    self.newp2State = torch.zeros(18)

	-- CvC specific variables
    self.xState = torch.zeros(18)
    self.Q = torch.zeros(9)
    self.normQ = nn.SoftMax()

    --loop until game ends
    while true do

		-- process x turn and evaluate
		self:comTurn(xTurn,AI)
        self:evaluateBoard()
		
		-- commit xTurn to memory
		self:selfEval_p1(AI)
		self:oppEval_p2(AI)

        --finish game if terminal
        if self.xWin or self.oWin or self.tie then
            break
        end

		-- process o turn and evaluate
		self:comTurn(oTurn,AI)
        self:evaluateBoard()

		self:selfEval_p2(AI)
		self:oppEval_p1(AI)

        --finish game if terminal
        if self.xWin or self.oWin or self.tie then
            break
        end

	    --nextTurn
        self.turn = self.turn + 1

    end
end


function ticTacToe:comTurn(cTurn,AI)

	local besti
    local done = false

	local locState
	local locAction

	if cTurn==xTurn then
    	self.p1State = self.xState
		locState = self.p1State
	else
		oTempState = torch.zeros(18)
		oTempState[{{1,9}}] = self.xState[{{10,18}}]
    	oTempState[{{10,18}}] = self.xState[{{1,9}}]
		self.p2State = oTempState
		locState = self.p2State
	end

    repeat
        repeat
            --determine next move
            if torch.uniform() > AI.eps then				--exploit
				local temp = 1e2
                self.Q = self.normQ:forward(AI:process(locState)*temp)
                local spin = torch.uniform()
                for chance = 1,9 do
                    if spin<self.Q[{{1,chance}}]:sum() then
                        locAction = chance
                        break
                    end
                end
            else                            --explore
                besti = torch.randperm(9)
                locAction = besti[1]
            end
            --punish for invalid moves
            if self.xState[locAction]~=0 or self.xState[locAction+9]~=0 then
                AI.memIndex = AI.memIndex + 1
                AI.memory[AI.memIndex] = {locState, locState, locAction, self.invalidScore, false}
                break   --break inner loop
            end
			if cTurn== xTurn then
	            --update p1 state
    	        self.xState[locAction] = 1
        	    self.newp1State = self.xState:clone()
				self.p1State = locState
				self.p1Action = locAction
			else
	            --update p2 state
    	        self.xState[locAction+9] = 1
        	    self.newp2State[{{1,9}}] = self.xState[{{10,18}}]:clone()
            	self.newp2State[{{10,18}}] = self.xState[{{1,9}}]:clone()
				self.p2State = locState
				self.p2Action = locAction
			end
            done = true --exit outer loop
		until true
	until done

end



function ticTacToe:selfEval_p1(AI)
    if self.xWin then       --win
        AI.memIndex = AI.memIndex + 1
        AI.memory[AI.memIndex] = {self.p1State, self.newp1State:clone(), self.p1Action, self.winScore, true}
    elseif self.tie then    --tie
        AI.memIndex = AI.memIndex + 1
        AI.memory[AI.memIndex] = {self.p1State, self.newp1State:clone(), self.p1Action, self.tieScore, true}
    end
end



function ticTacToe:selfEval_p2(AI)
    if self.oWin then    --win
        AI.memIndex = AI.memIndex + 1
        AI.memory[AI.memIndex] = {self.p2State, self.newp2State:clone(), self.p2Action, self.winScore, true}
    elseif self.tie then --tie
        AI.memIndex = AI.memIndex + 1
        AI.memory[AI.memIndex] = {self.p2State, self.newp2State:clone(), self.p2Action, self.tieScore, true}
    end
end



function ticTacToe:oppEval_p2(AI)
    if self.turn > 1 then
        --update p2 state
        self.newp2State[{{1,9}}] = self.xState[{{10,18}}]
        self.newp2State[{{10,18}}] = self.xState[{{1,9}}]
        if self.xWin then    --lose
            AI.memIndex = AI.memIndex + 1
            AI.memory[AI.memIndex] = {self.p2State, self.newp2State:clone(), self.p2Action, self.loseScore, true}
        elseif self.tie then --tie
            AI.memIndex = AI.memIndex + 1
            AI.memory[AI.memIndex] = {self.p2State, self.newp2State:clone(), self.p2Action, self.tieScore, true}
        else        --uneventful
            AI.memIndex = AI.memIndex + 1
            AI.memory[AI.memIndex] = {self.p2State, self.newp2State:clone(), self.p2Action, self.noScore, false}
        end
    end
end

function ticTacToe:oppEval_p1(AI)
    --update p1 state
    self.newp1State = self.xState:clone()
    --p1 opponent eval
    if self.oWin then    --lose
        AI.memIndex = AI.memIndex + 1
        AI.memory[AI.memIndex] = {self.p1State, self.newp1State:clone(), self.p1Action, self.loseScore, true}
    elseif tie then --tie
        AI.memIndex = AI.memIndex + 1
        AI.memory[AI.memIndex] = {self.p1State, self.newp1State:clone(), self.p1Action, self.tieScore, true}
    else        --uneventful
        AI.memIndex = AI.memIndex + 1
        AI.memory[AI.memIndex] = {self.p1State, self.newp1State:clone(), self.p1Action, self.noScore, false}
    end
end



function ticTacToe:evaluateBoard()

    --x victory
    gameBoard = self.xState[{{1,9}}]
    for loop = 1,8 do
        self.xWin = evalVicType(loop, gameBoard)
    end

    --o victory
    gameBoard = self.xState[{{10,18}}]
    for loop = 1,8 do
        self.oWin = evalVicType(loop, gameBoard)
    end

    --tie game
    moves = self.xState:ne(0):sum()
    self.tie = moves==9

end



function evalVicType(i, gameBoard)
    t = 1;
    for j = 1,3 do
        t = t*gameBoard[victory[i][j]]
    end
    return t ~= 0
end

