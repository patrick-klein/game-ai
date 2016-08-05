--dependencies
require 'torch'
require 'nn'
--require 'cudatorch'	--eventually, maybe

-------------------learn-------------------------
function learn(net, originalState, nextState,
               action, reward, terminal)

    --eta, etaPlus, etaMinus, delta0, alpha, maxIter, maxBatches, lambda, magicNumber, autocorr, rigidity
    
    gamma = 0.05
    
    originalQ = net:forward(originalState)
    
    y = originalQ:clone()
    if terminal then
        y[action] = reward
    else
        yMax = net:forward(nextState)
        yMax = yMax:max()
        y[action] = reward + gamma*yMax
    end

    criterion = nn.MSECriterion()

    for loop = 1,15 do
        criterion:forward(net:forward(originalState), y)
        net:zeroGradParameters()
        gradOutput = criterion:backward(net.output, y)
        net:backward(originalState, gradOutput)
        net:updateParameters(0.01)
    end

    return net

end
