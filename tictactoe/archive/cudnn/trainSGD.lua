--dependencies
require 'torch'
require 'nn'

-------------------learn-------------------------
function trainSGD(net, dataset)

    criterion = nn.MSECriterion():cuda()
    trainer = nn.StochasticGradient(net, criterion)
    trainer.learningRate = 1e-6
    trainer.verbose = false
    trainer.maxIteration = 1
    trainer.shuffleIndices = false
    trainer:train(dataset)

    return net

end
