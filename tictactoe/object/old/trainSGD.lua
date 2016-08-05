require 'torch'
require 'nn'

require 'cutorch'
require 'cunn'

function trainSGD(net, trainSet, moves)

    --get globals, otherwise set to default
    gamma = gamma or 0.5
    learnRate = learnRate or 0.001
    alpha = alpha or 0.1

    data = {}
    function data:size() return moves end

    for move = 1,moves do

        --read values from set
        local origState = trainSet[move][1]
        local nextState = trainSet[move][2]
        local action = trainSet[move][3]
        local reward = trainSet[move][4]
        local terminal = trainSet[move][5]

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
        local target = output:clone()
        target[action] = (1-alpha)*output[action]+alpha*y

        data[move] = {origState, target}
    
    end

    criterion = nn.MSECriterion()
    trainer = nn.StochasticGradient(net, criterion)
    trainer.learningRate = learnRate
    trainer.verbose = false
    trainer:train(data)

    return net

end
