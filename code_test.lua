--
-- Created by IntelliJ IDEA.
-- User: aabdujyo
-- Date: 18/02/17
-- Time: 5:30 PM
-- To change this template use File | Settings | File Templates.
--

require 'nn'
require 'nngraph'

n_channels = 1
rnn_size = 256

x = nn.Identity()()
x_error_prev = nn.Identity()()

layer1 = nn.SpatialConvolution(n_channels, 12, 5, 5, 2, 2)(x)
layer1 = nn.ReLU()(layer1)
layer2 = nn.SpatialConvolution(12, 16, 3, 3)(layer1)
layer2 = nn.ReLU()(layer2)
layer3 = nn.SpatialConvolution(16, 16, 3, 3)(layer2)
layer3 = nn.ReLU()(layer3)
layer3_flat = nn.View(16*8*8)(layer3)
fc1 = nn.Linear(16*8*8, rnn_size)(layer3_flat)

--net = nn.gModule({x}, {fc1})


layer1_e = nn.SpatialConvolution(n_channels, 12, 5, 5, 2, 2)(x_error_prev)
layer1_e = nn.ReLU()(layer1_e)
layer2_e = nn.SpatialConvolution(12, 16, 3, 3)(layer1_e)
layer2_e = nn.ReLU()(layer2_e)
layer3_e = nn.SpatialConvolution(16, 16, 3, 3)(layer2_e)
layer3_e = nn.ReLU()(layer3_e)
layer3_flat_e = nn.View(16*8*8)(layer3_e)
fc1_e = nn.Linear(16*8*8, rnn_size)(layer3_flat)

out = nn.JoinTable(2)({fc1, fc1_e})
out = nn.View(rnn_size)(out)

net = nn.gModule({x, x_error_prev}, {fc1, fc1_e, out})

input = torch.rand(1, 28, 28)
input2 = torch.rand(1, 28, 28)

f1, f1e, output = net:forward({input, input2})
--output = net:forward(input)

print(output)



