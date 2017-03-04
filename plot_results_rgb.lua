--require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'
require 'cutorch'
local model_utils=require 'model_utils'
local mnist = require 'mnist'
nngraph.setDebug(true)

--x_prediction = torch.load('t9/x_prediction')
x_prediction = torch.load('t9/x_generation')

--print(#x_prediction)
--print(x_prediction[1][1]:size())

x = torch.zeros(#x_prediction, x_prediction[1][1]:size(2), x_prediction[1][1]:size(3), x_prediction[1][1]:size(4))
--x = torch.zeros(#x_prediction, x_prediction[1]:size(2), x_prediction[1]:size(3))
x = x:cuda()
for i = 1, x_prediction[1][1]:size(1) do
    for t = 1, #x_prediction do
        x[{{t}, {}, {}, {}}] = x_prediction[t][1][i]:cuda()
        --x[{{t}, {}, {}}] = x_prediction[t][i]:cuda()
    end
    image.display(x)
end