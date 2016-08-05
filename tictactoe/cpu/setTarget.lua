
-------------------learn-------------------------
function setTarget(net, dataset)

    --discount factor
    gamma = gamma or 0.5
    
    --set target
    local y = torch.zeros(9)
    if terminal then
        y[action] = reward
    else
        local originalQ = net:forward(originalState):clone()
        local yMax = originalQ:max()
        y[action] = reward + gamma*yMax
    end

    -- can probably get rid of clones, using local variables
    dataset[dataIndex] = {originalState:clone(), y:clone()}

    return target

end
