--dependencies
require 'torch'
require 'nn'

-------------------learn-------------------------
function train(net, dataset)

    criterion = nn.MSECriterion()
    trainer = nn.StochasticGradient(net, criterion)
    trainer.learningRate = 0.001
    trainer.verbose = false
    trainer.maxIteration = 15
    trainer:train(dataset)

    return net

end
