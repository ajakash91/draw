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
n_data = 10

x = nn.Identity()()
--[[
--Convolution
x_error_prev = nn.Identity()()

layer1 = nn.SpatialConvolution(n_channels, 12, 5, 5, 1, 1)(x)
layer1 = nn.ReLU()(layer1)
layer2 = nn.SpatialConvolution(12, 16, 3, 3)(layer1)
layer2 = nn.ReLU()(layer2)
layer3 = nn.SpatialConvolution(16, 32, 3, 3, 2, 2)(layer2)
layer3 = nn.ReLU()(layer3)
layer3_flat = nn.View(32*10*10)(layer3)
fc1 = nn.Linear(32*10*10, rnn_size)(layer3_flat)

--net = nn.gModule({x}, {fc1})

layer1_e = nn.SpatialConvolution(n_channels, 12, 5, 5, 1, 1)(x_error_prev)
layer1_e = nn.ReLU(True)(layer1_e)
layer2_e = nn.SpatialConvolution(12, 16, 3, 3)(layer1_e)
layer2_e = nn.ReLU(True)(layer2_e)
layer3_e = nn.SpatialConvolution(16, 32, 3, 3, 2, 2)(layer2_e)
layer3_e = nn.ReLU(True)(layer3_e)
layer3_flat_e = nn.View(32*10*10)(layer3_e)
fc1_e = nn.Linear(32*10*10, rnn_size)(layer3_flat_e)

out = nn.JoinTable(2)({fc1, fc1_e})
out = nn.View(rnn_size)(out)

net = nn.gModule({x, x_error_prev}, {out})

input = torch.rand(n_data, 1, 28, 28)
input2 = torch.rand(n_data, 1, 28, 28)

output = net:forward({input, input2})
--output = net:forward(input)
]]--

-- Deconvolution
next_h = nn.Identity()()

fc_1 = nn.Linear(rnn_size, 32*10*10)(next_h)
fc_1 = nn.View(32, 10, 10)(fc_1)
fc_1 = nn.ReLU()(fc_1)
layer_1 = nn.SpatialFullConvolution(32, 16, 3, 3, 2, 2, 0, 0, 1, 1)(fc_1)
layer_1 = nn.ReLU()(layer_1)
layer_2 = nn.SpatialFullConvolution(16, 12, 3, 3)(layer_1)
layer_2 = nn.ReLU()(layer_2)
layer_3 = nn.SpatialFullConvolution(12, n_channels, 5, 5)(layer_2)

next_canvas = layer_3--nn.CAddTable()({prev_canvas, write_layer})

mu = nn.Sigmoid()(next_canvas)

neg_mu = nn.MulConstant(-1)(mu)
d = nn.CAddTable()({x, neg_mu})
d2 = nn.Power(2)(d)
loss_x = nn.Sum(4)(d2)
loss_x = nn.Sum(3)(loss_x)
loss_x = nn.Sum(2)(loss_x)

net = nn.gModule({x, next_h}, {next_canvas, loss_x})

--net = nn.gModule({next_h}, {next_canvas})

x = torch.rand(n_data, n_channels, 28, 28)
h = torch.rand(n_data, 256)

output = net:forward({x, h})
print(output)



