
--[[
  class for optimizing hyper-parameters of qLearner

  --keys in params must return a table of values, e.g.
  --    params = {numberOfLayers={2,3,4},
  --              numHiddenNodes={256,512,1024},
  --              ...,
  --              }

  -- more param options?
  --  include dropout
  --  backup
  --  reload

  -- meta options?
  --  load/save memory
  --  load pretrained network & finetune?

  Class Methods
  ...

]]

-- require libraries
require 'torch'
require 'nn'
require 'optim'
require 'io'

-- require classes
require 'AI/AI'
require 'AI/qLearner'

--globals


-- declare metaQLearner class
local metaQLearner, parent = torch.class('metaQLearner', 'AI')


-- initialization function
-- input should be table for parameters
function metaQLearner:__init(game, params)

  parent.__init(self, game)

  --assert params for initialization
  assert(params, 'No hyperparameters passed to metaQLearner.  Use qLearner instance if only default values desired.')
  self.params = params

  self.bestLearner = {}
  self.bestScore = 0
  self.bestConfig = {}

end


--method for finding optimal hyperparameters
function metaQLearner:train()

  --initialize gridCombo to {1,1,1,...} and store number of vals per param
  numParams = table.getn(self.params)
  gridCombo = {}
  numValsForParams = {}
  for param_key, param_vals in pairs(self.params) do
    gridCombo[param_key] = 1
    numValsForParams[param_key] = table.getn(self.params[param_key])
    last_param = param_key
  end

  --loop through all combos
  gridIndex = 0
  while true do

    --keep track of index
    gridIndex = gridIndex+1

    --set thisConfig from gridCombo
    thisConfig = {}
    for param_key, param_vals in pairs(self.params) do
      thisConfig[param_key] = param_vals[gridCombo[param_key]]
    end
    print('Iteration... '..gridIndex)
    --print(thisConfig)

    --initialize qLearner with config (and set optional params)
    thisLearner = qLearner(self.game.new(), thisConfig)
    thisLearner.saveMemory = index==1 --save memory on first iteration, otherwise load memory
    thisLearner.loadMemory = index~=1
    thisLearner.verbose = false
    thisLearner:train()
    thisScore = self:learnerEvaluate(thisLearner)

    --save learner/score/config if best
    if thisScore>self.bestScore then
      --can't just clone learner, so technically creating new one with same config & net.
      --this means it doesn't have the same replay memory, optimstate, etc.
      thisNet = thisLearner.net:clone()
      self.bestLearner = thisLearner.new(self.game.new(), thisConfig, thisNet)
      self.bestConfig = thisConfig
      self.bestScore = thisScore
    end

    --update gridCombo for next iteration, and check for main break condition
    for param_key, param_index in pairs(gridCombo) do
      --update index for this parameter if not at end of values
      if gridCombo[param_key]<numValsForParams[param_key] then
        gridCombo[param_key] = param_index + 1
        break
      --if last param and can't incrememnt, set break condition
      elseif param_key==last_param then
        breakCondition = true
      --reset index so next param can be incrememnted
      else
        gridCombo[param_key] = 1
      end
    end

    --check for breakCondition to exit while-do loop
    if breakCondition then
      break
    end

  end

  --print best config
  print('Optimal hyperparameters:')
  print(self.bestConfig)

end


function metaQLearner:learnerEvaluate(learner)

  local numTrials = 1e2
  local runningTotal = 0
  for myEval = 1, numTrials do
    torch.manualSeed(42 * myEval)
    runningTotal = runningTotal + learner.game:test()
  end

  torch.seed()
  if self.game.name == '2048' then
    score = runningTotal / numTrials
  elseif self.game.name == 'TicTacToe' then
    score = (1 / 2) * (1 + runningTotal / numTrials)
  end

  return score

end


-- method for running test trials
function metaQLearner:selfEvaluate()
  assert(false, 'metaQLearner:selfEvaluate() method not implemented!')
  return self.bestLearner:selfEvaluate()  --leaving behind assert, because is this even needed?
end


--public shorthand for forward pass model
function metaQLearner:process(input)
  return self.bestLearner:process(input)
end


function metaQLearner:save()
  --Note: despite file name, saved object has class qLearner
  torch.save('saves/metaQLearner_'..self.game.name..'.ai', self.bestLearner)
end
