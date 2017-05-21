
-- require libraries
require 'torch'
require 'nn'
--require 'cutorch'
--require 'cunn'

require 'io'

-- require classes
require 'AI_cpu'

-- create game2048
local game2048 = torch.class('game2048')

-- initialization function
function game2048:__init()

    -- set scores
	self.newHighScore = 1
	self.gameOverScore = -1

	self.numInputs = 16
	self.numOutputs = 4

	self.gameType = 'single-player'

	self.AI = nil
	self.testmode = false

end


function game2048:play(player)


	--input argument checks
	assert(player==hum or player==com,'Computer or human player required.')

	--initialize board states 
	self:generateBoard()
	prevState = self.state:view(16)

	local highScore = 0
	self.turn = 1
	self.draw = self.draw or player==hum
	local isTerminal
	local action
	local actionIndex
	local score = 0
	local memState

    --loop until game ends
    while true do

		if self.draw then
			self:drawBoard()
			os.execute('sleep 0.1')
		end

		--choose move
		-- valid inputs: 'w' 'a' 's' 'd'
		if player==hum then action = self:playerTurn() end
		if player==com then action = self:comTurn() end


		--update state
		self.state = self:updateBoard(action)

		self:insertValue()
		memState = self.state:view(16)

		if torch.max(self.state)>highScore then
			highScore = torch.max(self.state)
			--score = self.newHighScore
		--else
			--score = 0
		end
		isTerminal = self:gameOver()

		--create some memories
		if self.AI and not self.testmode then
			--score = self.didMerge and 1 or 0
			--score = isTerminal and -1 or self.mergeSum
			score = isTerminal and -1 or (self.didMerge and 1 or 0)
			--score = isTerminal and -1 or score
			actionIndex = self:getMoveIndexFromString(action)
			self.AI.memIndex = self.AI.memIndex + 1
			self.AI.memory[self.AI.memIndex] = {prevState, memState, actionIndex, score, isTerminal}
			prevState = memState
		end


		--check for terminal condition
		if isTerminal then
			--if self.testmode then self:drawBoard() end
			if self.draw then
				self:drawBoard()
				print('Game Over!')
				print(self.turn+1)
			end
			return highScore
		end

		self.turn = self.turn+1
	end
end


--public method for running trials on AI
function game2048:test()
	self.testmode = true
	score = self:play(com)
	self.testmode = false
	return score
end

--private method, adds random values to initial board
---- consider making this more general, by adding random values to available blank squares
function game2048:generateBoard()

	self.state = torch.Tensor(4,4):zero()

	self:insertValue()
	self:insertValue()

end


--private method, shorthand for choosing 2 or 4 randomly
function game2048:getNewRandTileVal()
	return 2*(1+torch.round(torch.uniform()))
end


--privdate method, updates board according to action
function game2048:updateBoard(action)

	local newState = self.state:clone()

	--define local function to rotate board
	local function rotateCW90(rotate_by)
		local tempArray = torch.Tensor(4,4):zero()
		if rotate_by==-1 then newState=newState:t() end
		for row = 1,4 do
			for col = 1,4 do
				tempArray[4-row+1][col] = newState[row][col]
			end
		end
		newState = tempArray
		if rotate_by==1 then newState=newState:t() end
		newState = newState:contiguous()
	end
	

	--rotate so shifting is in the upwards direction
	--lua doesn't have a switch statement... *sigh*
	if action=='a' then
		rotateCW90(1)
	elseif action=='s' then
		rotateCW90(-1)
		rotateCW90(-1)
	elseif action=='d' then
		rotateCW90(-1)
	end

	self.didMerge = false
	self.mergeSum = 0
	self.numMerge = 0

	--shift everything upwards
	local nextCol=false local ignoreMerge=false
	for col = 1,4 do
		for row = 1,3 do
			ignoreMerge = false
			for rowUphill = row+1,4 do
				-- if square is zero, and non-zero somewhere below, shift to that square and move on
				if newState[row][col]==0 and newState[rowUphill][col]~=0 then
					newState[row][col] = newState[rowUphill][col]
					newState[rowUphill][col] = 0
				end
				-- if next non-zero block is same, merge (unless already merged)
				if newState[rowUphill][col]==newState[row][col] and newState[row][col]~=0 and not ignoreMerge then
					newState[row][col] = newState[row][col]*2
					self.mergeSum = self.mergeSum + newState[row][col]/1024
					newState[rowUphill][col] = 0
					ignoreMerge = true
					self.didMerge = true
					break
				-- if non-zero somewhere below, shift under this square
				elseif newState[rowUphill][col]~=0 then
					if rowUphill>row+1 then
						newState[row+1][col] = newState[rowUphill][col]
						newState[rowUphill][col] = 0
					end
					ignoreMerge = false
					break
				end
				-- if empty below, just skip to next column
				if rowUphill==4 then nextCol = true end
			end
			if nextCol then
				nextCol = false
				break
			end
		end
	end

	--rotate back to original orientation
	if action=='a' then
		rotateCW90(-1)
	elseif action=='s' then
		rotateCW90(-1)
		rotateCW90(-1)
	elseif action=='d' then
		rotateCW90(1)
	end

	return newState

end


--private method, insert new value into empty tile
function game2048:insertValue()

	local tempArray = torch.zeros(16)
	local tempArrayIndex = 0

	self.state:view(self.state,16)

	for index=1,16 do
		if self.state[index]==0 then
			tempArrayIndex = tempArrayIndex+1
			tempArray[tempArrayIndex] = index
		end
	end

	if tempArrayIndex==0 then print(self.state:view(4,4)) end

	local randArray = torch.randperm(tempArrayIndex)

	self.state[tempArray[randArray[1]]] = self:getNewRandTileVal()
	self.state:view(self.state,4,4)

end



--private method, gets valid input from player
function game2048:playerTurn()
	repeat
		action = io.read()
		isValidKey = action=='w' or action=='a' or action=='s' or action=='d'
		if isValidKey then
			isValidMove = not torch.all(torch.eq(self.state, self:updateBoard(action)))
		end
	until isValidKey and isValidMove
	return action
end

--private method, generates move from the AI
function game2048:comTurn()

	--use eps value if provided by AI
	local eps = 0 or self.AI.eps

	--AI processed move
	if self.AI and (torch.uniform()>eps or self.testmode) then
		local Q = self.AI:process(self.state:view(16))
		Qsorted, Qindices = torch.sort(Q,1,true)
		actionList = Qindices
		--if Qsorted[1]>5 and self.testmode then print(Qsorted[1]) end
		--print(Qsorted[1])
	else
		actionList = torch.randperm(4)
	end
	for i = 1,4 do
		action = self:getMoveStringFromIndex(actionList[i])
       	isValidMove = not torch.all(torch.eq(self.state, self:updateBoard(action)))
       	if isValidMove then return action end
    end
end

--private method, converts move indices into strings
function game2048:getMoveStringFromIndex(index)
    if index==1 then return 'w'
    elseif index==2 then return 'a'
    elseif index==3 then return 's'
    elseif index==4 then return 'd'
	end
end


--private method, converts move string to index
function game2048:getMoveIndexFromString(string)
    if string=='w' then return 1
    elseif string=='a' then return 2
    elseif string=='s' then return 3
    elseif string=='d' then return 4
	end
end


--private method, checks is there are any valid moves left
function game2048:gameOver()
	for i=1,4 do
		action = self:getMoveStringFromIndex(i)
		if not torch.all(torch.eq(self.state, self:updateBoard(action))) then
			return false
		end
	end
	return true
end


--private method, draws board
function game2048:drawBoard()

	for row = 1,4 do

		io.write('\n\t')
		io.write(self.state[row][1])
		io.write('\t')
		io.write(self.state[row][2])
		io.write('\t')
		io.write(self.state[row][3])
		io.write('\t')
		io.write(self.state[row][4])

	end

	io.write('\n\n')

end
