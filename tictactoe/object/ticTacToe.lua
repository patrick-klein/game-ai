-- require libraries
require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'

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


function ticTacToe:play(p1,p2)

    --initial board states
    self.p1State = torch.zeros(18):cuda()
    self.p2State = torch.zeros(18):cuda()
    self.p1StateNew = torch.zeros(18)
    self.p2StateNew = torch.zeros(18)

    -- CvC specific variables
    self.xState = torch.zeros(18)
    self.Q = torch.zeros(9)
    self.normQ = nn.SoftMax():cuda()

	self.turn = 1

    --loop until game ends
    while true do

		-- moves need to update xState
		if p1 == com then self:comTurn(xTurn)
		elseif p1 == challenge then self:randTurn(xTurn)
		elseif p1 == hum then self:playerTurn(xTurn)
		end

       	self:evaluateBoard()

		--commit xTurn to memory
		--(always? even with challenge or player? force AI, even for pvp?)
       	self:selfEval(xTurn)
       	self:oppEval(oTurn)

        --finish game if terminal
        if self.xWin or self.oWin or self.tie then break
		end

		if p2 == com then self:comTurn(oTurn)
		elseif p2 == challenge then self:randTurn(oTurn)
		elseif p2 == hum then self:playerTurn(oTurn)
		end

        self:evaluateBoard()

        self:selfEval(oTurn)
        self:oppEval(xTurn)

        --finish game if terminal
        if self.xWin or self.oWin or self.tie then break
        end

        --nextTurn
        self.turn = self.turn + 1

    end

	if p1==challenge then
		if self.xWin then return lose
		elseif self.oWin then return win
		elseif self.tie then return draw
		end
	elseif p2==challenge then
		if self.xWin then return win
		elseif self.oWin then return lose
		elseif self.tie then return draw
		end
	end

end


function ticTacToe:comTurn(cTurn)

	local besti
    local done = false

	local locState
	local locAction

	if cTurn==xTurn then
    	self.p1State = self.xState:cuda()
		locState = self.p1State

	else
		local oTempState = torch.zeros(18)
		oTempState[{{1,9}}] = self.xState[{{10,18}}]
    	oTempState[{{10,18}}] = self.xState[{{1,9}}]
		self.p2State = oTempState:cuda()
		locState = self.p2State
	end

	local temp = 1

    repeat
        repeat
            --determine next move
            if torch.uniform() > self.AI.eps then				--exploit
                self.Q = self.normQ:forward(self.AI:process(locState)*temp)
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
                self.AI.memIndex = self.AI.memIndex + 1
                self.AI.memory[self.AI.memIndex] = {locState:float(), locState:float(), locAction, self.invalidScore, false}
                break   --break inner loop
            end

			if cTurn== xTurn then
	            --update p1 state
    	        self.xState[locAction] = 1
        	    self.p1StateNew = self.xState:clone()
				self.p1State = locState
				self.p1Action = locAction

			else
	            --update p2 state
    	        self.xState[locAction+9] = 1
        	    self.p2StateNew[{{1,9}}] = self.xState[{{10,18}}]:clone()
            	self.p2StateNew[{{10,18}}] = self.xState[{{1,9}}]:clone()
				self.p2State = locState
				self.p2Action = locAction
			end

            done = true --exit outer loop
		until true
	until done

end

function ticTacToe:randTurn(rTurn)

	local locSort = torch.randperm(9)
	for locLoop=1,9 do
		--repeat until valid move
		if self.xState[locSort[locLoop]]==0 and self.xState[9+locSort[locLoop]]==0 then
			if rTurn == xTurn then
				self.xState[locSort[locLoop]] = 1
			elseif rTurn == oTurn then
				self.xState[9+locSort[locLoop]] = 1
			end
			break   --break inner loop
		end
	end

end

function ticTacToe:playerTurn(pTurn)

end


function ticTacToe:selfEval(player)

	local p1 = xTurn
	local p2 = oTurn

	assert(player==p1 or player==p2, 'Unrecognized player passed to ticTacToe:selfEval')

	local pState
	local pStateNew
	local pAction

	if player==p1  then
		pState = self.p1State:float()
		pStateNew = self.p1StateNew:clone()
		pAction = self.p1Action
	elseif player==p2 then
		pState = self.p2State:float()
		pStateNew = self.p2StateNew:clone()
		pAction = self.p2Action
	end

	assert(pStateNew, 'Nil value passed to pStateNew in selfEval')

    if (p1 and self.xWin) or (p2 and self.oWin) then       --win
        self.AI.memIndex = self.AI.memIndex + 1
        self.AI.memory[self.AI.memIndex] = {pState, pStateNew, pAction, self.winScore, true}

    elseif self.tie then    --tie
        self.AI.memIndex = self.AI.memIndex + 1
        self.AI.memory[self.AI.memIndex] = {pState, pStateNew, pAction, self.tieScore, true}
    end

end

function ticTacToe:oppEval(player)

	local p1 = xTurn
	local p2 = oTurn

	assert(player==p1 or player==p2, 'Unrecognized player passed to ticTacToe:oppEval')

    if p1 or (self.turn>1 and p2) then

		local pState
		local pStateNew
		local pAction

		if p1 then		--update p1 state
			self.p1StateNew = self.xState:clone()
			pState = self.p1State:float()
			pStateNew = self.p1StateNew:clone()
			pAction = self.p1Action

		elseif p2 then	--update p2 state
	        self.p2StateNew[{{1,9}}] = self.xState[{{10,18}}]
    	    self.p2StateNew[{{10,18}}] = self.xState[{{1,9}}]
			pState = self.p2State:float()
			pStateNew = self.p2StateNew:clone()
			pAction = self.p2Action
		end

		assert(pStateNew, 'Nil value passed to pStateNew in oppEval')

        if (p1 and self.oWin) or (p2 and self.xWin) then	--lose
            self.AI.memIndex = self.AI.memIndex + 1
            self.AI.memory[self.AI.memIndex] = {pState, pStateNew, pAction, self.loseScore, true}

        elseif self.tie then	--tie
            self.AI.memIndex = self.AI.memIndex + 1
            self.AI.memory[self.AI.memIndex] = {pState, pStateNew, pAction, self.tieScore, true}

        else	--uneventful
            self.AI.memIndex = self.AI.memIndex + 1
            self.AI.memory[self.AI.memIndex] = {pState, pStateNew, pAction, self.noScore, false}
        end
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

