--[[
  Implementation of the mobile game 2048

  Class Methods
    __init()
    play(player)
    test()
    generateBoard()
    getNewRandTileVal()
    updateBoard(actionIndex)
    insertValue()
    playerTurn()
    comTurn()
    gameOver()
    drawBoard()
    augmentMemory(prevState, memState, actionIndex, score, isTerminal)

  local functions
    rotateCW90(board, rotate_by)
    scaleDownLog2(input)
    expandBoard(input)

]]

--require libraries
require 'torch'
require 'nn'
require 'io'

--require classes
require 'games/game'
require 'AI/AI'

--create class
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

  --draw by default for humans
  self.draw = self.draw or player == hum

  --declare and initialize variables
  local highScore = 0
  self.turn = 1
  local isTerminal
  local action = nil
  local actionIndex
  local score = 0

  --loop until game ends
  while true do

    --draws board
    if self.draw then
      self:drawBoard(action)
      os.execute('sleep 0.1')
    end

    --call method to get move from player or AI
    if player == hum then
      actionIndex = self:playerTurn()
    elseif player == com then
      actionIndex = self:comTurn()
    end

    --store state for memory
    local prevState = self.state:clone()

    --update state
    self.state = self:updateBoard(actionIndex)

    --randomly insert value into board
    self:insertValue()

    --clone state to be used in replay memory
    ----should the state be captured before OR after insertValue?
    local memState = self.state:clone()

    --assign score which will be returned
    if torch.max(self.state) > highScore then
      highScore = torch.max(self.state)
    end

    --check for terminal board state
    isTerminal = self:gameOver()

    --create some memories (and augment) and pass to AI
    if self.AI and not self.testmode then
      score = isTerminal and -1 or 0
      self:augmentMemory(prevState, memState, actionIndex, score, isTerminal)
    end

    --end game if terminal
    if isTerminal then

      --draw final move
      if self.draw then
        self:drawBoard(action)
        print('Game Over!')
        print(self.turn + 1)
        print(highScore)
      end

      --return store
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


--define local function to rotate board
local function rotateCW90(board, rotate_by)
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
  return newState
end


--scales down input/scores from 2,4,8,16... to 1,2,3,4...
local function scaleDownLog2(input)
  return torch.floor(torch.log1p(input) / torch.log(2))
end


--expands indexed 4x4 board into scaled and flattened 4x4x12 board
local function expandBoard(input)

  local oldBoard = scaleDownLog2(input:clone())
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
  return newBoard

end


--privdate method, returns new board according to action
function twenty48:updateBoard(actionIndex)

  newState = self.state:clone()

  --get rotated copy of board so shifting is in the upwards direction
  if actionIndex == 2 then
    newState = rotateCW90(newState, -1)
  elseif actionIndex == 3 then
    newState = rotateCW90(newState, -1)
    newState = rotateCW90(newState, -1)
  elseif actionIndex == 4 then
    newState = rotateCW90(newState, 1)
  end

  --initialize variables to remember if values merged
  --(used in reward)
  self.didMerge = false

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
  if actionIndex == 2 then
    newState = rotateCW90(newState, 1)
  elseif actionIndex == 3 then
    newState = rotateCW90(newState, -1)
    newState = rotateCW90(newState, -1)
  elseif actionIndex == 4 then
    newState = rotateCW90(newState, -1)
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
  actionTable = {w=1, d=2, s=3, a=4}
  --keep trying until valid input received
  repeat
    actionString = io.read()
    actionIndex = actionTable[actionString]
    isValidKey = actionIndex ~= nil
    if isValidKey then
      isValidMove = not torch.all(torch.eq(self.state, self:updateBoard(actionIndex)))
    end
  until isValidKey and isValidMove
  return actionIndex
end

--private method, generates move from the AI
function twenty48:comTurn()

  --use eps value if provided by AI
  local eps = 0 or self.AI.eps

  --generate moves using AI or randomly
  if self.AI and (torch.uniform() > eps or self.testmode) then
    local Q = self.AI:process(expandBoard(self.state))
    Qsorted, Qindices = torch.sort(Q, 1, true)
    actionList = Qindices
  else
    actionList = torch.randperm(4)
  end

  --check if moves are valid, return if true
  for i = 1, 4 do

    actionIndex = actionList[i]
    isValidMove = not torch.all(torch.eq(self.state, self:updateBoard(actionIndex)))

    if isValidMove then
      return actionIndex

    -- penalize incorrect inputs
    elseif not self.testmode then
      self:augmentMemory(self.state, self.state, actionIndex, -1, false)

    end
  end

end




--private method, checks is there are any valid moves left
function twenty48:gameOver()
  --loop through all possible moves, terminal if board is still full
  for i = 1, 4 do
    gameOverCondition = torch.all(torch.eq(self.state, self:updateBoard(i)))
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


local function rotateAction(actionIndex, rotateBy)
  return ((actionIndex-1+rotateBy)%4)+1
end


--save move to replay memory (including augmentations)
function twenty48:augmentMemory(prevState, memState, actionIndex, score, isTerminal)

  --actual move
  prevState_s = expandBoard(prevState)
  memState_s = expandBoard(memState)
  self.AI.memIndex = self.AI.memIndex + 1
  self.AI.memory[self.AI.memIndex] = {prevState_s, memState_s, actionIndex, score, isTerminal}

  --rotate 90
  actionIndex_90 = rotateAction(actionIndex, 1)
  prevState_90 = expandBoard(rotateCW90(prevState, 1))
  memState_90 = expandBoard(rotateCW90(memState, 1))
  self.AI.memIndex = self.AI.memIndex + 1
  self.AI.memory[self.AI.memIndex] = {prevState_90, memState_90, actionIndex_90, score, isTerminal}

  --rotate 180
  actionIndex_180 = rotateAction(actionIndex, 2)
  prevState_180 = expandBoard( rotateCW90( rotateCW90(prevState, -1), -1))
  memState_180 = expandBoard( rotateCW90( rotateCW90(memState, -1), -1))
  self.AI.memIndex = self.AI.memIndex + 1
  self.AI.memory[self.AI.memIndex] = {prevState_180, memState_180, actionIndex_180, score, isTerminal}

  --rotate 270
  actionIndex_270 = rotateAction(actionIndex, 3)
  prevState_270 = expandBoard(rotateCW90(prevState, -1))
  memState_270 = expandBoard(rotateCW90(memState, -1))
  self.AI.memIndex = self.AI.memIndex + 1
  self.AI.memory[self.AI.memIndex] = {prevState_270, memState_270, actionIndex_270, score, isTerminal}

  --flip lr
  --flip ud

end
