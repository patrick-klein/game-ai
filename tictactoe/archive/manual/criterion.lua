--dependencies
require 'torch'
require 'nn'
--require 'cudatorch'	--eventually, maybe

local QLearningCriterion, parent = torch.class('nn.QLearningCriterion', 'nn.Criterion')

function QLearningCriterion:__init(originalState, newState, action, reward, terminal, gamma)
    parent.__init(self)
    
    if originalState and newState and action and reward and terminal then
        self.originalState = originalState
        self.newState = newState
        self.action = action
        self.reward = reward
        self.terminal = terminal
    else
        error("Six input arguments expected.")
    end

    gamma = gamma or 0.01
    self.gamma = gamma

end



function QLearningCriterion:updateOutput(input, target)


    --loss
    self.output = target-input:

    if input:dim() == 1 and input:size()[1] == 2 then
        self.output = -input[target] 
        if target == 2 then
            self.output = self.output * self.scaleFactor
        end
    else
        error('vector of 2 elements expected')
    end
    return self.output



end



function QLearningCriterion:updateGradInput(input, target)

    

    self.gradInput:resizeAs(input)
    self.gradInput:zero()
    
    self.gradInput[target] = -1
    if target == 2 then
        self.gradInput[target] = self.gradInput[target] * self.scaleFactor
    end
    
    return self.gradInput

end
