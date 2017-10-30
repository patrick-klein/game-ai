--[[
  Implementation of the mobile game 2048

  Class Methods
    __init()
    play(player)
    test()
    generateBoard()
    getNewRandTileVal()
    updateBoard(action)
    insertValue()
    playerTurn()
    comTurn()
    getMoveStringFromIndex(index)
    getMoveIndexFromString(string)
    gameOver()
    drawBoard()
    scaleDownLog2(input)
    expandBoard(input)

]]

-- require libraries
require 'torch'
require 'nn'
require 'io'

-- require classes
require 'games/game'
require 'AI/AI'

-- create twenty48
local twenty48, parent = torch.class('twenty48', 'game')

-- initialization function
function twenty48:__init()

  parent.__init(self)

  --set scores for rewards
  self.newHighScore = 1
  self.gameOverScore = -1

  self.name = '2048'

  self.draw = nil

  --used for plotting training progress
  --self.maxScore = 12
  self.maxScore = 512

  --needed for AI to function
  self.numInputs = 192
  self.numOutputs = 4
  self.numPlayers = 1

end

--public method that plays a game, player can be human or AI
function twenty48:play(player)

  --input argument checks
  assert(player == hum or player == com, 'Computer or human player required.')

  --initialize board states
  self:generateBoard()

  --declare and initialize variables
  local highScore = 0
  self.turn = 1
  self.draw = self.draw or player == hum
  local isTerminal
  local action = nil
  local actionIndex
  local score = 0

  --loop until game ends
  while true do

    --draws board
    --default on for human, off for AI
    if self.draw then
      self:drawBoard(action)
      os.execute('sleep 0.1')
    end

    -- call method to get move from player or AI
    if player == hum then action = self:playerTurn() end
    if player == com then action = self:comTurn() end

    local prevState = self:expandBoard(self.state)

    --update state
    self.state = self:updateBoard(action)

    --randomly insert value into board
    self:insertValue()

    --clone state to be used in replay memory
    ----should the state be captured before OR after insertValue?
    local memState = self:expandBoard(self.state)

    --DEBUG--
    if false then
      print(prevState:view(4, 4))
      print(memState:view(4, 4))
      print(action)
      assert(false)
    end

    --assign score which will be returned
    if torch.max(self.state) > highScore then
      highScore = torch.max(self.state)
    end

    --check for terminal board state
    isTerminal = self:gameOver()

    --create some memories and pass to AI
    if self.AI and not self.testmode then
      score = isTerminal and - 1 or self.mergeSum
      actionIndex = self:getMoveIndexFromString(action)
      self.AI.memIndex = self.AI.memIndex + 1
      --it's a good idea to scale back the input logarithmically
      self.AI.memory[self.AI.memIndex] = {prevState, memState, actionIndex, score, isTerminal}
      --prevState = memState:clone()
    end

    --end game if terminal
    if isTerminal then
      if self.draw then
        self:drawBoard(action)
        print('Game Over!')
        print(self.turn + 1)
        print(highScore)
      end
      --return highScore, scaled down
      --return self:scaleDownLog2(highScore)
      return self.turn + 1
    end

    --increment turn counter
    self.turn = self.turn + 1
  end
end


--public method for running trials on AI
function twenty48:test()
  self.testmode = true
  score = self:play(com)
  self.testmode = false
  return score
end


--private method, adds random values to initial board
function twenty48:generateBoard()
  self.state = torch.Tensor(4, 4):zero()
  self:insertValue()
  self:insertValue()
end


--private method, shorthand for choosing 2 or 4 randomly
function twenty48:getNewRandTileVal()
  return 2 * (1 + torch.round(torch.uniform()))
end


--privdate method, returns new board according to action
function twenty48:updateBoard(action)

  --copy board while manipulating it
  local newState = self.state:clone()

  --define local function to rotate board
  local function rotateCW90(rotate_by)
    local tempArray = torch.Tensor(4, 4):zero()
    if rotate_by == -1 then newState = newState:t() end
    for row = 1, 4 do
      for col = 1, 4 do
        tempArray[4 - row + 1][col] = newState[row][col]
      end
    end
    newState = tempArray
    if rotate_by == 1 then newState = newState:t() end
    newState = newState:contiguous()
  end

  --rotate so shifting is in the upwards direction
  ----lua doesn't have a switch statement... *sigh*
  if action == 'a' then
    rotateCW90(1)
  elseif action == 's' then
    rotateCW90(-1)
    rotateCW90(-1)
  elseif action == 'd' then
    rotateCW90(-1)
  end

  --initialize some more variables
  self.didMerge = false
  self.mergeSum = 0
  self.numMerge = 0

  --shift everything upwards
  local nextCol = false local ignoreMerge = false
  for col = 1, 4 do
    for row = 1, 3 do
      ignoreMerge = false
      for rowUphill = row + 1, 4 do
        -- if square is zero, and non-zero somewhere below, shift to that square and move on
        if newState[row][col] == 0 and newState[rowUphill][col] ~= 0 then
          newState[row][col] = newState[rowUphill][col]
          newState[rowUphill][col] = 0
        end
        -- if next non-zero block is same, merge (unless already merged)
        if newState[rowUphill][col] == newState[row][col] and newState[row][col] ~= 0 and not ignoreMerge then
          newState[row][col] = newState[row][col] * 2
          self.mergeSum = self.mergeSum + (self:scaleDownLog2(newState[row][col]) / 11)^2
          newState[rowUphill][col] = 0
          ignoreMerge = true
          self.didMerge = true
          break
          -- if non-zero somewhere below, shift under this square
        elseif newState[rowUphill][col] ~= 0 then
          if rowUphill > row + 1 then
            newState[row + 1][col] = newState[rowUphill][col]
            newState[rowUphill][col] = 0
          end
          ignoreMerge = false
          break
        end
        -- if empty below, just skip to next column
        if rowUphill == 4 then nextCol = true end
      end
      if nextCol then
        nextCol = false
        break
      end
    end
  end

  --rotate back to original orientation
  if action == 'a' then
    rotateCW90(-1)
  elseif action == 's' then
    rotateCW90(-1)
    rotateCW90(-1)
  elseif action == 'd' then
    rotateCW90(1)
  end

  --return new board
  return newState

end


--private method, insert new value into empty tile
function twenty48:insertValue()

  --flatten board, create new vector to store zero indicies
  local tempArray = torch.zeros(16)
  local tempArrayIndex = 0
  self.state:view(self.state, 16)

  --find all zero indices
  for index = 1, 16 do
    if self.state[index] == 0 then
      tempArrayIndex = tempArrayIndex + 1
      tempArray[tempArrayIndex] = index
    end
  end

  --shuffle indices
  local randArray = torch.randperm(tempArrayIndex)

  --randomly insert 2 or 4 into empty tile
  self.state[tempArray[randArray[1]]] = self:getNewRandTileVal()

  --unflatten board
  self.state:view(self.state, 4, 4)

end



--private method, gets valid input from player (w,a,s,d)
function twenty48:playerTurn()
  --keep trying until valid input received
  repeat
    action = io.read()
    isValidKey = action == 'w' or action == 'a' or action == 's' or action == 'd'
    if isValidKey then
      isValidMove = not torch.all(torch.eq(self.state, self:updateBoard(action)))
    end
  until isValidKey and isValidMove
  return action
end

--private method, generates move from the AI
function twenty48:comTurn()

  --use eps value if provided by AI
  local eps = 0 or self.AI.eps

  --generate moves using AI or randomly
  if self.AI and (torch.uniform() > eps or self.testmode) then
    local Q = self.AI:process(self:expandBoard(self.state))
    Qsorted, Qindices = torch.sort(Q, 1, true)
    actionList = Qindices
  else
    actionList = torch.randperm(4)
  end

  --check if moves are valid, return if true
  for i = 1, 4 do
    actionIndex = actionList[i]
    actionString = self:getMoveStringFromIndex(actionIndex)
    isValidMove = not torch.all(torch.eq(self.state, self:updateBoard(actionString)))
    if isValidMove then
      return actionString
    elseif not self.testmode then
      -- penalize incorrect inputs
      self.AI.memIndex = self.AI.memIndex + 1
      self.AI.memory[self.AI.memIndex] = {self:expandBoard(self.state), self:expandBoard(self.state), actionIndex, - 1, false}
    end
  end
end


--private method, converts move indices into strings
----could probably just use typedef or tuple (or lua equivalent)
function twenty48:getMoveStringFromIndex(index)
  if index == 1 then return 'w'
  elseif index == 2 then return 'a'
  elseif index == 3 then return 's'
  elseif index == 4 then return 'd'
  end
end


--private method, converts move string to index
function twenty48:getMoveIndexFromString(string)
  if string == 'w' then return 1
  elseif string == 'a' then return 2
  elseif string == 's' then return 3
  elseif string == 'd' then return 4
  end
end


--private method, checks is there are any valid moves left
function twenty48:gameOver()
  --loop through all possible moves, terminal if board is still full
  for i = 1, 4 do
    action = self:getMoveStringFromIndex(i)
    local storedMergeSum = self.mergeSum
    gameOverCondition = torch.all(torch.eq(self.state, self:updateBoard(action)))
    self.mergeSum = storedMergeSum
    if not gameOverCondition then
      return false
    end
  end
  return true
end


--private method, draws board
function twenty48:drawBoard(action)

  if action == 'w' then io.write('↑')
  elseif action == 'a' then io.write('←')
  elseif action == 's' then io.write('↓')
  elseif action == 'd' then io.write('→')
  end

  io.write('\n')

  --draw each row separately
  for row = 1, 4 do
    io.write('\n\t')
    if self.state[row][1] > 0 then
      io.write(self.state[row][1])
    else io.write('-') end
    io.write('\t')
    if self.state[row][2] > 0 then
      io.write(self.state[row][2])
    else io.write('-') end
    io.write('\t')
    if self.state[row][3] > 0 then
      io.write(self.state[row][3])
    else io.write('-') end
    io.write('\t')
    if self.state[row][4] > 0 then
      io.write(self.state[row][4])
    else io.write('-') end
  end
  io.write('\n')

end


--scales down input/scores from 2,4,8,16... to 1,2,3,4...
function twenty48:scaleDownLog2(input)
  return torch.floor(torch.log1p(input) / torch.log(2))
end


--expands indexed 4x4 board into scaled and flattened 4x4x12 board
function twenty48:expandBoard(input)

  local oldBoard = self:scaleDownLog2(input:clone())
  local newBoard = torch.Tensor(4, 4, 12):zero()

  for row = 1, 4 do
    for col = 1, 4 do
      if oldBoard[row][col] > 0 then
        if oldBoard[row][col] <= 12 then
          newBoard[row][col][oldBoard[row][col]] = 1
        else
          newBoard[row][col][12] = 1
        end
      end
    end
  end

  newBoard = newBoard:view(192)
  --newBoard = (newBoard:view(192) - .5) / .5

  return newBoard

end
