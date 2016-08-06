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

	--set new parameters
	self.p1 = p1
	self.p2 = p2

	self.isCom = p1==com or p2==com
	self.isChallenge = p1==challenge or p2==challenge
	self.isHum = p1==hum or p2==hum

	assert(not(isHum or isCom),'Computer or human player required.')

    --initial board states (1 is X, -1 is O)
	--don't need to keep on GPU, only move when processed
    self.state = torch.zeros(9)
    self.prevState = torch.zeros(9)
	self.action = 0
	
	--AI input is relative to player (1 is player, -1 is opponent)

	self.xWin = false
	self.oWin = false
	self.tie = false

	if p1==hum then self:drawBoard() end

	self.turn = 1

    --loop until game ends
    while true do

		--moves need to update state,prevState,action
		if p1 == com then self:comTurn(xTurn)
		elseif p1 == challenge then self:randTurn(xTurn)
		elseif p1 == hum then self:playerTurn(xTurn)
		end

		if self.turn>=3 then self:evaluateBoard() end
		if isHum then self:drawBoard() end

		if not isChallenge then
       		self:selfEval(xTurn)
       		self:oppEval(oTurn)
		end

        --finish game if terminal
        if self.xWin or self.oWin or self.tie then break
		end

		if p2 == com then self:comTurn(oTurn)
		elseif p2 == challenge then self:randTurn(oTurn)
		elseif p2 == hum then self:playerTurn(oTurn)
		end

		if self.turn>=3 then self:evaluateBoard() end
		if isHum then self:drawBoard() end

		if not isChallenge then
        	self:selfEval(oTurn)
        	self:oppEval(xTurn)
		end

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

	--X gets state, O gets negative state
	local pState = cTurn==xTurn and self.state:clone() or -self.state
	local pAction

    local Q = torch.zeros(9)
    local normQ = nn.SoftMax()

	local randMoves
    local done = false
	local first = true

	local temp = isHum and 10 or 1
	local eps = not (isHum or isChallenge) and self.AI.eps or 0.

	--double repeat loops replicate continue/break logic
    repeat
        repeat
            --determine next move
            if torch.uniform() > eps then	--exploit
				if first then		--only process state once
	                Q = normQ:cuda():forward(self.AI:process(pState:cuda())*temp)
					first = false
				else				--subsequent tries ignore invalid move, renormalize
					Q[pAction] = 0
					Q = normQ:forward(Q)
				end

                local spin = torch.uniform()
                for chance = 1,9 do
                    if spin<Q[{{1,chance}}]:sum() then
                        pAction = chance
                        break
                    end
                end

            else                            --explore
                randMoves = torch.randperm(9)
                pAction = randMoves[1]
            end

            --check move validity
            if self.state[pAction]~=0 then
				if not isChallenge then	--remember if not challenge
	                self.AI.memIndex = self.AI.memIndex + 1
    	            self.AI.memory[self.AI.memIndex] = {pState, pState, pAction, self.invalidScore, false}
				end
				break   	--continue
			end

			--update boards
			self.prevState = self.state:clone()
			self.state[pAction] = cTurn==xTurn and 1 or -1
			self.action = pAction

            done = true		--break
		until true
	until done

end

function ticTacToe:randTurn(rTurn)

	local randMoves = torch.randperm(9)
	local pAction

	for move=1,9 do
		--repeat until valid move
		pAction = randMoves[move]
		if self.state[pAction]==0 then
			self.prevState = self.state:clone()				-- probably not necessary, memories not saved
			self.state[pAction] = rTurn==xTurn and 1 or -1
			self.action = pAction
			break
		end
	end

end

function ticTacToe:playerTurn(pTurn)
	
	local pAction

	repeat	--until valid move is input
		pAction = io.read();
		if self.state[pAction]==0 then
			self.prevState = self.state:clone()
			self.state[pAction] = pTurn==xTurn and 1 or -1
			self.action = pAction
			break
		end
	until false

end


function ticTacToe:selfEval(player)

	assert(player==xTurn or player==oTurn, 'Unrecognized player passed to ticTacToe:selfEval')

	local pStatePrev = player==xTurn and self.prevState:clone() or -self.prevState
	local pState = player==xTurn and self.state:clone() or -self.state
	local pAction = self.action

    if (xTurn and self.xWin) or (oTurn and self.oWin) then       --win
        self.AI.memIndex = self.AI.memIndex + 1
        self.AI.memory[self.AI.memIndex] = {pStatePrev, pState, pAction, self.winScore, true}

    elseif self.tie then    --tie
        self.AI.memIndex = self.AI.memIndex + 1
        self.AI.memory[self.AI.memIndex] = {pStatePrev, pState, pAction, self.tieScore, true}
    end

end

function ticTacToe:oppEval(player)

	--debug
	assert(player==xTurn or player==oTurn, 'Unrecognized player passed to ticTacToe:oppEval')

    if xTurn or (self.turn>1 and oTurn) then

		--set local relative states
		local pStatePrev = player==xTurn and self.prevState:clone() or -self.prevState
		local pState = player==xTurn and self.state:clone() or -self.state
		local pAction = self.action

		--save lose,tie,uneventful memory
        if (xTurn and self.oWin) or (oTurn and self.xWin) then	--lose
            self.AI.memIndex = self.AI.memIndex + 1
            self.AI.memory[self.AI.memIndex] = {pStatePrev, pState, pAction, self.loseScore, true}

        elseif self.tie then	--tie
            self.AI.memIndex = self.AI.memIndex + 1
            self.AI.memory[self.AI.memIndex] = {pStatePrev, pState, pAction, self.tieScore, true}

        else	--uneventful
            self.AI.memIndex = self.AI.memIndex + 1
            self.AI.memory[self.AI.memIndex] = {pStatePrev, pState, pAction, self.noScore, false}
        end
    end
end


function ticTacToe:evaluateBoard()

	local vicType

    --x victory
	local xState = self.state:eq(1)	
    for vicType = 1,8 do
        self.xWin = self.xWin or evalVicType(vicType, xState)
    end

    --o victory
	local oState = self.state:eq(-1)	
    for vicType = 1,8 do
        self.oWin = self.oWin or evalVicType(vicType, oState)
    end

    --tie game
	local numMoves = self.state:ne(0):sum()
    self.tie = numMoves==9

end


function evalVicType(i, gameBoard)
    t = 1;
    for j = 1,3 do
        t = t*gameBoard[victory[i][j]]
    end
    return t~=0
end

function ticTacToe:drawBoard()

	xBoard = self.state:clone()
	oBoard = -self.state

    io.write("\t")

    if xBoard[1] == 1 then io.write('x')
    elseif oBoard[1] == 1 then io.write('o')
    else io.write('.') end

    io.write("\t|\t")

    if xBoard[2] == 1 then io.write('x')
    elseif oBoard[2] == 1 then io.write('o')
    else io.write('.') end

    io.write("\t|\t")

    if xBoard[3] == 1 then io.write('x')
    elseif oBoard[3] == 1 then io.write('o')
    else io.write('.') end

    io.write("\n\t")

    if xBoard[4] == 1 then io.write('x')
    elseif oBoard[4] == 1 then io.write('o')
    else io.write('.') end

    io.write("\t|\t")

    if xBoard[5] == 1 then io.write('x')
    elseif oBoard[5] == 1 then io.write('o')
    else io.write('.') end

    io.write("\t|\t")

    if xBoard[6] == 1 then io.write('x')
    elseif oBoard[6] == 1 then io.write('o')
    else io.write('.') end

    io.write("\n\t")

    if xBoard[7] == 1 then io.write('x')
    elseif oBoard[7] == 1 then io.write('o')
    else io.write('.') end

    io.write("\t|\t")

    if xBoard[8] == 1 then io.write('x')
    elseif oBoard[8] == 1 then io.write('o')
    else io.write('.') end

    io.write("\t|\t")

    if xBoard[9] == 1 then io.write('x')
    elseif oBoard[9] == 1 then io.write('o')
    else io.write('.') end

	io.write('\n\n\n')

end
