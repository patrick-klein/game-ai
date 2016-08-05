
-------------------learn-------------------------
function remember(origState, nextState, action,
                  reward, terminal, dataSet, dataIndex)

    dataSet[dataIndex] = {origState:clone(), nextState:clone(), action, reward, terminal}

    return dataSet

end
