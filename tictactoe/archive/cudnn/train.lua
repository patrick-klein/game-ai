function train(net, trainSet)

    gamma = gamma or 0.5
    learnRate = learnRate or 0.001

    local origState = trainSet[1]
    local nextState = trainSet[2]
    local action = trainSet[3]
    local reward = trainSet[4]
    local terminal = trainSet[5]

    if terminal then
        y = reward
    else
        local Qnext = net:forward(nextState)
        y = reward + gamma*Qnext:max()
    end

    local target = torch.zeros(9):cuda()
    target[action] = y

    local output = net:forward(origState)
    
    local gradOut = torch.zeros(9):cuda()
    gradOut[action] = -target[action]+output[action]
    
    net:zeroGradParameters()
    net:backward(origState, gradOut)
    net:updateParameters(learnRate)

    return net

end
