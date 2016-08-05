--dependencies
require 'torch'
require 'nn'

-------------------learn-------------------------
function remember(net, originalState, nextState,
                  action, reward, terminal, dataset, dataIndex)

    gamma = 0.1
    
    y = net:forward(originalState):clone()

    if terminal then
        y[action] = reward
    else
        yMax = net:forward(nextState)
        yMax = yMax:max()
        y[action] = reward + gamma*yMax
    end

    dataset[dataIndex] = {originalState:cuda(), y:cuda()}

    return dataset

end
