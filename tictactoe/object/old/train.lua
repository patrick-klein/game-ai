require 'torch'
require 'nn'

function train(net, trainSet)

    --get globals, otherwise set to default
    gamma = gamma or 0.5
    learnRate = learnRate or 0.001
    alpha = alpha or 0.1

    --read values from set
    local origState = trainSet[1]
    local nextState = trainSet[2]
    local action = trainSet[3]
    local reward = trainSet[4]
    local terminal = trainSet[5]

    --set desired Q value for action
    if terminal then                            --terminal gets reward only
        y = reward
    else                                        --non-terminal adds future (discounted) reward
        local Qnext = net:forward(nextState)    --calculate expected return using current parameters
        y = reward + gamma*Qnext:max()
    end

    --calculate Q using current parameters
    local output = net:forward(origState)

    --set target to vector
    --local target = torch.zeros(9)
    local target = output:clone()
    target[action] = (1-alpha)*output[action]+alpha*y
    --target[action] = y
    
    --find difference between current Q and expected return
    --local gradOut = torch.zeros(9)
    --gradOut[action] = target[action]-output[action]
    criterion = nn.MSECriterion()
    criterion.sizeAverage = false
    criterion:forward(net:forward(origState), target)
    gradOut = criterion:backward(net.output, target)
    
    --one gradient descent step (for this move only)
    net:zeroGradParameters()
    net:backward(origState, gradOut)
    net:updateParameters(learnRate)

    return net

end
