--[[

  Class Methods
    __init()
    play(p1,p2)
    test()
    comTurn(cTurn)
    randTurn(rTurn)
    playerTurn(pTurn)
    selfEval(player)
    oppEval(player)
    evaluateBoard()
    drawBoard()

]]

-- require libraries
require 'torch'
require 'nn'
require 'io'

-- require classes
require 'AI/AI'

-- create ticTacToe_class
local ticTacToe, parent = torch.class('ticTacToe', 'game')

-- local constants
xTurn = 1
oTurn = 2
challenge = 3
lose = -1
win = 1
draw = 0

-- victory types
victory = torch.Tensor(8, 3);
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

  parent.__init(self)

  -- set scores
  self.winScore = 1
  self.loseScore = -1
  self.invalidScore = -1
  self.tieScore = 0
  self.noScore = 0

  self.name = 'TicTacToe'

  self.numInputs = 9
  self.numOutputs = 9
  self.numPlayers = 2

  self.draw = nil

  self.maxScore = 1

end


function ticTacToe:play(p1, p2)

  --set new parameters
  self.p1 = p1
  self.p2 = p2

  self.isCom = p1 == com or p2 == com
  self.isChallenge = p1 == challenge or p2 == challenge
  self.isHum = p1 == hum or p2 == hum

  assert(p1 and p2, 'Two players are require for Tic Tac Toe.')
  assert(self.isHum or self.isCom, 'Computer or human player required.')

  if self.isCom then assert(self.AI, 'AI needed to play game.') end

  --initial board states (1 is X, -1 is O)
  self.state = torch.zeros(9)
  self.prevStateX = torch.zeros(9)
  self.prevStateO = torch.zeros(9)
  self.actionX = 0
  self.actionO = 0

  --AI input is relative to player (1 is player, -1 is opponent)

  self.xWin = false
  self.oWin = false
  self.tie = false

  self.draw = self.draw or (p1 == hum or p2 == hum)
  if self.draw then self:drawBoard() end

  self.turn = 1

  --loop until game ends
  while true do

    --moves need to update state,prevState,action
    if p1 == com then self:comTurn(xTurn)
    elseif p1 == challenge then self:randTurn(xTurn)
    elseif p1 == hum then self:playerTurn(xTurn)
    end

    if self.turn >= 3 then self:evaluateBoard() end
    if self.draw then self:drawBoard() end

    if not self.isChallenge then
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

    if self.turn >= 3 then self:evaluateBoard() end
    if self.draw then self:drawBoard() end

    if not self.isChallenge then
      self:selfEval(oTurn)
      self:oppEval(xTurn)
    end

    --finish game if terminal
    if self.xWin or self.oWin or self.tie then break
    end

    --nextTurn
    self.turn = self.turn + 1

  end

  if self.draw then
    if self.xWin then print('X Wins!')
    elseif self.oWin then print('O Wins!')
    elseif self.tie then print('Draw!')
    end
  end

  if p1 == challenge then
    if self.xWin then return lose
    elseif self.oWin then return win
    elseif self.tie then return draw
    end
  elseif p2 == challenge then
    if self.xWin then return win
    elseif self.oWin then return lose
    elseif self.tie then return draw
    end
  end
end


--public method for running trials on AI
function ticTacToe:test()
  if torch.uniform() > 0.5 then
    return self:play(com, challenge)
  else
    return self:play(challenge, com)
  end
end


function ticTacToe:comTurn(cTurn)

  --X gets state, O gets negative state
  local pState = cTurn == xTurn and self.state:clone() or - self.state
  local pAction

  local Q = torch.zeros(9)
  local normQ = nn.SoftMax()
  local spin

  local randMoves
  local randAttempt = 0

  local done = false
  local first = true

  --local exploit = false
  local exploit = (self.isChallenge or self.isHum)
  local explore = false

  --local temp = (self.isChallenge or self.isHum) and 100 or 1
  --local eps = (self.isChallenge or self.isHum) and 0 or self.AI.eps

  local temp = 10
  local eps = self.AI.eps or 0

  --double repeat loops replicate continue/break logic
  repeat
    repeat
      --determine next move
      if (torch.uniform() > eps or exploit) and not explore then --exploit
        if first then --only process state once
          Q = normQ:forward(self.AI:process(pState * temp))
          exploit = true
          first = false
        else --subsequent tries ignore invalid move, renormalize
          Q[pAction] = 0
          Q = normQ:forward(Q)
        end
        spin = torch.uniform()
        for chance = 1, 9 do
          if spin < Q[{{1, chance}}]:sum() then
            pAction = chance
            break
          end
        end
      else --explore
        if randAttempt == 0 then
          randMoves = torch.randperm(9)
          explore = true
        end
        randAttempt = randAttempt + 1
        pAction = randMoves[randAttempt]
      end

      --check move validity
      if self.state[pAction] ~= 0 then
        if not self.isChallenge then --remember if not challenge
          self.AI.memIndex = self.AI.memIndex + 1
          self.AI.memory[self.AI.memIndex] = {pState, pState, pAction, self.invalidScore, false}
        end
        break --continue
      end

      if self.isHum and exploit then print(Q[pAction]) end

      --update boards
      if cTurn == xTurn then
        self.prevStateX = self.state:clone()
        self.actionX = pAction
        self.state[pAction] = 1
      elseif cTurn == oTurn then
        self.prevStateO = self.state:clone()
        self.actionO = pAction
        self.state[pAction] = -1
      end

      done = true --break
    until true
  until done
end


function ticTacToe:randTurn(rTurn)

  local randMoves = torch.randperm(9)
  local pAction

  for move = 1, 9 do --repeat until valid move
    pAction = randMoves[move]
    if self.state[pAction] == 0 then
      --update boards
      if rTurn == xTurn then
        self.prevStateX = self.state:clone()
        self.actionX = pAction
        self.state[pAction] = 1
      elseif rTurn == oTurn then
        self.prevStateO = self.state:clone()
        self.actionO = pAction
        self.state[pAction] = -1
      end
      break
    end
  end

end


function ticTacToe:playerTurn(pTurn)

  local pAction

  repeat --until valid move is input
    pAction = tonumber(io.read());
    if pAction then
      if pAction >= 1 and pAction <= 9 then
        if self.state[pAction] == 0 then
          --update boards
          if pTurn == xTurn then
            self.prevStateX = self.state:clone()
            self.actionX = pAction
            self.state[pAction] = 1
          elseif pTurn == oTurn then
            self.prevStateO = self.state:clone()
            self.actionO = pAction
            self.state[pAction] = -1
          end
          break
        end
      end
    end
  until false

end


function ticTacToe:selfEval(player)

  assert(player == xTurn or player == oTurn, 'Unrecognized player passed to ticTacToe:selfEval')

  local pStatePrev
  local pState
  local pAction

  if player == xTurn then
    pStatePrev = self.prevStateX:clone()
    pState = self.state:clone()
    pAction = self.actionX
  elseif player == oTurn then
    pStatePrev = -self.prevStateO
    pState = -self.state
    pAction = self.actionO
  end

  if (player == xTurn and self.xWin) or (player == oTurn and self.oWin) then --win
    self.AI.memIndex = self.AI.memIndex + 1
    self.AI.memory[self.AI.memIndex] = {pStatePrev, pState, pAction, self.winScore, true}
  elseif self.tie then --tie
    self.AI.memIndex = self.AI.memIndex + 1
    self.AI.memory[self.AI.memIndex] = {pStatePrev, pState, pAction, self.tieScore, true}
  end

end


function ticTacToe:oppEval(player)

  --debug
  assert(player == xTurn or player == oTurn, 'Unrecognized player passed to ticTacToe:oppEval')

  if player == xTurn or (self.turn > 1 and player == oTurn) then

    local pStatePrev
    local pState
    local pAction

    if player == xTurn then
      pStatePrev = self.prevStateX:clone()
      pState = self.state:clone()
      pAction = self.actionX
    elseif player == oTurn then
      pStatePrev = -self.prevStateO
      pState = -self.state
      pAction = self.actionO
    end

    --save lose,tie,uneventful memory
    if (player == xTurn and self.oWin) or (player == oTurn and self.xWin) then --lose
      self.AI.memIndex = self.AI.memIndex + 1
      self.AI.memory[self.AI.memIndex] = {pStatePrev, pState, pAction, self.loseScore, true}
    elseif self.tie then --tie
      self.AI.memIndex = self.AI.memIndex + 1
      self.AI.memory[self.AI.memIndex] = {pStatePrev, pState, pAction, self.tieScore, true}
    else --uneventful
      self.AI.memIndex = self.AI.memIndex + 1
      self.AI.memory[self.AI.memIndex] = {pStatePrev, pState, pAction, self.noScore, false}
    end
  end
end


function ticTacToe:evaluateBoard()

  local vicType

  local function evalVicType(i, gameBoard)
    t = 1;
    for j = 1, 3 do
      t = t * gameBoard[victory[i][j]]
    end
    return t ~= 0
  end

  --x victory
  local xState = self.state:eq(1)
  for vicType = 1, 8 do
    self.xWin = self.xWin or evalVicType(vicType, xState)
  end

  --o victory
  local oState = self.state:eq(-1)
  for vicType = 1, 8 do
    self.oWin = self.oWin or evalVicType(vicType, oState)
  end

  --tie game
  local numMoves = self.state:ne(0):sum()
  self.tie = numMoves == 9

end


function ticTacToe:drawBoard()

  xBoard = self.state:eq(1)
  oBoard = self.state:eq(-1)

  local function drawSquare(sqr)
    if xBoard[sqr] == 1 then io.write('x')
    elseif oBoard[sqr] == 1 then io.write('o')
    else io.write('.') end
  end

  --draw each row separately
  for row = 0, 2 do
    io.write('\n\t')
    drawSquare(3 * row + 1)
    io.write('\t|\t')
    drawSquare(3 * row + 2)
    io.write('\t|\t')
    drawSquare(3 * row + 3)
  end
  io.write('\n\n')

end
