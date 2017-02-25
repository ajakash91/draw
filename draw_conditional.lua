--require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'
local model_utils=require 'model_utils'
local mnist = require 'mnist'
require 'cutorch'
require 'cunn'

--nngraph.setDebug(true)

Tensor = torch.CudaTensor

n_z = 100			--20    --400
rnn_size = 256		--100   --1024
n_canvas = 28*28
seq_length = 15		--50
n_channels = 3

--N = 15				--3
-- Image Height
A = 28
-- Image Width
B = 28
n_data = 30
--encoder 
x = nn.Identity()()
x_error_prev = nn.Identity()()
dir_name = ""
file_extention = '.t7'

o1 = 48
o2 = 96
o3 = 128
f1 = 3
f2 = 2
f3 = 2
final_width = A-f1-f2-f3+3

--[[function enc_convolution(x)
    layer1 = nn.SpatialConvolution(n_channels, o1, f1, f1)(x)
    layer1 = nn.ReLU()(layer1)
    layer2 = nn.SpatialConvolution(o1, o2, f2, f2)(layer1)
    layer2 = nn.ReLU()(layer2)
    layer3_flat = nn.View(o2*(A-f1-f2+2)*(A-f1-f2+2))(layer2)
    fc = nn.Linear(o2*(A-f1-f2+2)*(A-f1-f2+2), rnn_size)(layer3_flat)

    return(fc)
end]]--

--[[
f = torch.randperm(150)
for i = 1, 10 do
    fi[i] = files[f[i]%]
end
fi

as = string.match(fi[7], '%d%d%d%..*%.t7') -- 190.Red_cockaded_Woodpecker.t7
as = as:sub(1, -4) --190.Red_cockaded_Woodpecker

]]--

function read_data(dir_name, file_extention)
    -- Go over all files in directory. We use an iterator, paths.files().
    for file in paths.files(dir_name) do
       -- We only load files that match the extension
       if file:find(file_extention .. '$') then
          -- and insert the ones we care about in our table
          table.insert(files, paths.concat(dir_name,file))
       end
    end

    -- Check files
    if #files == 0 then
       error('given directory doesnt contain any files of type: ' .. opt.ext)
    end

    -- Sorting the files
    table.sort(files, function (a,b) return a < b end)

    local text_features = {}
    for i,file in ipairs(files) do
       -- load each image
       table.insert(text_features, torch.load(file))
    end
    --print(features)

    -- Read images using the filenames
    local images =

    return images, text_features
end

function enc_convolution(x)
    layer1 = nn.SpatialConvolution(n_channels, o1, f1, f1)(x)
    layer1 = nn.ReLU()(layer1)
    layer2 = nn.SpatialConvolution(o1, o2, f2, f2)(layer1)
    layer2 = nn.ReLU()(layer2)
    layer3 = nn.SpatialConvolution(o2, o3, f3, f3)(layer2)
    layer3 = nn.ReLU()(layer3)
    layer3_flat = nn.View(o3*(final_width)*(final_width))(layer3)
    fc = nn.Linear(o3*(final_width)*(final_width), rnn_size)(layer3_flat)

    return(fc)
end

--read
--[[layer1 = nn.SpatialConvolution(n_channels, 12, 5, 5, 1, 1)(x)
layer1 = nn.ReLU()(layer1)
layer2 = nn.SpatialConvolution(12, 16, 3, 3)(layer1)
layer2 = nn.ReLU()(layer2)
layer3 = nn.SpatialConvolution(16, 32, 3, 3, 2, 2)(layer2)
layer3 = nn.ReLU()(layer3)
layer3_flat = nn.View(32*10*10)(layer3)
fc1 = nn.Linear(32*10*10, rnn_size)(layer3_flat)

layer1_e = nn.SpatialConvolution(n_channels, 12, 5, 5, 1, 1)(x_error_prev)
layer1_e = nn.ReLU(True)(layer1_e)
layer2_e = nn.SpatialConvolution(12, 16, 3, 3)(layer1_e)
layer2_e = nn.ReLU(True)(layer2_e)
layer3_e = nn.SpatialConvolution(16, 32, 3, 3, 2, 2)(layer2_e)
layer3_e = nn.ReLU(True)(layer3_e)
layer3_flat_e = nn.View(32*10*10)(layer3_e)
fc1_e = nn.Linear(32*10*10, rnn_size)(layer3_flat_e)]]--

--[[layer1 = nn.SpatialConvolution(n_channels, 16, 5, 5)(x)
layer1 = nn.ReLU()(layer1)
layer2 = nn.SpatialConvolution(16, 32, 5, 5)(layer1)
layer2 = nn.ReLU()(layer2)
layer3 = nn.SpatialConvolution(32, 64, 5, 5)(layer2)
layer3 = nn.ReLU()(layer3)
layer4 = nn.SpatialConvolution(64, 128, 5, 5)(layer3)
layer4 = nn.ReLU()(layer4)
layer4_flat = nn.View(128*12*12)(layer4)
fc1 = nn.Linear(128*12*12, rnn_size)(layer4_flat)

layer1_e = nn.SpatialConvolution(n_channels, 16, 5, 5)(x_error_prev)
layer1_e = nn.ReLU()(layer1_e)
layer2_e = nn.SpatialConvolution(16, 32, 5, 5)(layer1_e)
layer2_e = nn.ReLU()(layer2_e)
layer3_e = nn.SpatialConvolution(32, 64, 5, 5)(layer2_e)
layer3_e = nn.ReLU()(layer3_e)
layer4_e = nn.SpatialConvolution(64, 128, 5, 5)(layer3_e)
layer4_e = nn.ReLU()(layer4_e)
layer4_flat_e = nn.View(128*12*12)(layer4_e)
fc1_e = nn.Linear(128*12*12, rnn_size)(layer4_flat_e)
]]--

--[[layer1 = nn.SpatialConvolution(n_channels, 32, 5, 5)(x)
layer1 = nn.ReLU()(layer1)
layer2 = nn.SpatialConvolution(32, 64, 5, 5)(layer1)
layer2 = nn.ReLU()(layer2)
--layer3 = nn.SpatialConvolution(16, 32, 3, 3, 2, 2)(layer2)
--layer3 = nn.ReLU()(layer3)
layer3_flat = nn.View(64*20*20)(layer2)
fc1 = nn.Linear(64*20*20, rnn_size)(layer3_flat)

layer1_e = nn.SpatialConvolution(n_channels, 32, 5, 5)(x_error_prev)
layer1_e = nn.ReLU(True)(layer1_e)
layer2_e = nn.SpatialConvolution(32, 64, 5, 5)(layer1_e)
layer2_e = nn.ReLU(True)(layer2_e)
--layer3_e = nn.SpatialConvolution(16, 32, 3, 3, 2, 2)(layer2_e)
--layer3_e = nn.ReLU(True)(layer3_e)
layer3_flat_e = nn.View(64*20*20)(layer2_e)
fc1_e = nn.Linear(64*20*20, rnn_size)(layer3_flat_e)]]--

fc1 = enc_convolution(x)
fc1_e = enc_convolution(x_error_prev)

-- Add condition layer here. Rewrite

--read end

input = nn.JoinTable(2)({fc1, fc1_e})
input = nn.View(rnn_size*2)(input)
n_input = rnn_size*2

prev_h = nn.Identity()()
prev_c = nn.Identity()()

function new_input_sum()
    -- transforms input
    i2h            = nn.Linear(n_input, rnn_size)(input)
    -- transforms previous timestep's output
    h2h            = nn.Linear(rnn_size, rnn_size)(prev_h)
    return nn.CAddTable()({i2h, h2h})
end

in_gate          = nn.Sigmoid()(new_input_sum())
forget_gate      = nn.Sigmoid()(new_input_sum())
out_gate         = nn.Sigmoid()(new_input_sum())
in_transform     = nn.Tanh()(new_input_sum())

next_c           = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate,     in_transform})
})
next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

mu = nn.Linear(rnn_size, n_z)(next_h)
sigma = nn.Linear(rnn_size, n_z)(next_h)
sigma = nn.Exp()(sigma)

e = nn.Identity()()
sigma_e = nn.CMulTable()({sigma, e})
z = nn.CAddTable()({mu, sigma_e})
mu_squared = nn.Square()(mu)
sigma_squared = nn.Square()(sigma)
log_sigma_sq = nn.Log()(sigma_squared)
minus_log_sigma = nn.MulConstant(-1)(log_sigma_sq)
loss_z = nn.CAddTable()({mu_squared, sigma_squared, minus_log_sigma})
loss_z = nn.AddConstant(-1)(loss_z)
loss_z = nn.MulConstant(0.5)(loss_z)
loss_z = nn.Sum(2)(loss_z)
encoder = nn.gModule({x, x_error_prev, prev_c, prev_h, e}, {z, loss_z, next_c, next_h, patch})
encoder = encoder:cuda()
encoder.name = 'encoder'

--decoder
x = nn.Identity()()
z = nn.Identity()()
prev_h = nn.Identity()()
prev_c = nn.Identity()()
prev_canvas = nn.Identity()()
n_input = n_z
input = z

function new_input_sum()
    -- transforms input
    i2h            = nn.Linear(n_input, rnn_size)(input)
    -- transforms previous timestep's output
    h2h            = nn.Linear(rnn_size, rnn_size)(prev_h)
    return nn.CAddTable()({i2h, h2h})
end

in_gate          = nn.Sigmoid()(new_input_sum())
forget_gate      = nn.Sigmoid()(new_input_sum())
out_gate         = nn.Sigmoid()(new_input_sum())
in_transform     = nn.Tanh()(new_input_sum())

next_c           = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate,     in_transform})
})
next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})


-- write layer
--[[fc_1 = nn.Linear(rnn_size, 32*10*10)(next_h)
fc_1 = nn.View(32, 10, 10)(fc_1)
fc_1 = nn.ReLU()(fc_1)
layer_1 = nn.SpatialFullConvolution(32, 16, 3, 3, 2, 2, 0, 0, 1, 1)(fc_1)
layer_1 = nn.ReLU()(layer_1)
layer_2 = nn.SpatialFullConvolution(16, 12, 3, 3)(layer_1)
layer_2 = nn.ReLU()(layer_2)
layer_3 = nn.SpatialFullConvolution(12, n_channels, 5, 5)(layer_2)]]--

--[[fc_1 = nn.Linear(rnn_size, 128*12*12)(next_h)
fc_1 = nn.View(128, 12, 12)(fc_1)
fc_1 = nn.ReLU()(fc_1)
layer_1 = nn.SpatialFullConvolution(128, 64, 5, 5)(fc_1)
layer_1 = nn.ReLU()(layer_1)
layer_2 = nn.SpatialFullConvolution(64, 32, 5, 5)(layer_1)
layer_2 = nn.ReLU()(layer_2)
layer_3 = nn.SpatialFullConvolution(32, 16, 5, 5)(layer_2)
layer_3 = nn.ReLU()(layer_3)
layer_4 = nn.SpatialFullConvolution(16, n_channels, 5, 5)(layer_3)]]--

fc_1 = nn.Linear(rnn_size, o3*(final_width)*(final_width))(next_h)
fc_1 = nn.View(o3, (final_width), (final_width))(fc_1)
fc_1 = nn.ReLU()(fc_1)
layer_1 = nn.SpatialFullConvolution(o3, o2, f3, f3)(fc_1)
layer_1 = nn.ReLU()(layer_1)
layer_2 = nn.SpatialFullConvolution(o2, o1, f2, f2)(layer_1)
layer_2 = nn.ReLU()(layer_2)
layer_3 = nn.SpatialFullConvolution(o1, n_channels, f1, f1)(layer_2)

write_layer = layer_3

--write layer end

next_canvas = nn.CAddTable()({prev_canvas, write_layer})

mu = nn.Sigmoid()(next_canvas)

neg_mu = nn.MulConstant(-1)(mu)
d = nn.CAddTable()({x, neg_mu})
d2 = nn.Power(2)(d)
loss_x = nn.Sum(4)(d2)
loss_x = nn.Sum(3)(loss_x)
loss_x = nn.Sum(2)(loss_x)

x_prediction = nn.View(n_channels, A, B)(mu)
x_error = nn.View(n_channels, A, B)(d)

decoder = nn.gModule({x, z, prev_c, prev_h, prev_canvas}, {x_prediction, x_error, next_c, next_h, next_canvas, loss_x})
decoder = decoder:cuda()
decoder.name = 'decoder'

--train
trainset = mnist.traindataset()
testset = mnist.testdataset()

features_input = torch.zeros(n_data, n_channels, A, B)

--for i = 1, n_data do
    --features_input[{{i}, {1}, {}, {}}] = trainset[i].x:gt(125)
--end

images, text_features = read_data(dir_name, file_extention)


x = features_input
--print(x)
params, grad_params = model_utils.combine_all_parameters(encoder, decoder)

encoder_clones = model_utils.clone_many_times(encoder, seq_length)
decoder_clones = model_utils.clone_many_times(decoder, seq_length)

-- do fwd/bwd and return loss, grad_params
function feval(x_arg)
    if x_arg ~= params then
        params:copy(x_arg)
    end
    grad_params:zero()
    
    ------------------- forward pass -------------------
    lstm_c_enc = {[0]=torch.zeros(n_data, rnn_size)}
    lstm_h_enc = {[0]=torch.zeros(n_data, rnn_size)}
    lstm_c_dec = {[0]=torch.zeros(n_data, rnn_size)}
    lstm_h_dec = {[0]=torch.zeros(n_data, rnn_size)}
	

    x_error = {[0]=torch.rand(n_data, n_channels, A, B)}
    x_prediction = {}
    loss_z = {}
    loss_x = {}
    canvas = {[0]=torch.rand(n_data, n_channels, A, B)}
    x = {}
    --patch = {}
    
    local loss = 0

    for t = 1, seq_length do
      e[t] = torch.randn(n_data, n_z):cuda()
      x[t] = features_input:cuda()

	  lstm_h_enc[t-1] = lstm_h_enc[t-1]:cuda()
	  lstm_c_enc[t-1] = lstm_c_enc[t-1]:cuda()
      lstm_h_dec[t-1] = lstm_h_dec[t-1]:cuda()
	  lstm_c_dec[t-1] = lstm_c_dec[t-1]:cuda()
	  x_error[t-1] = x_error[t-1]:cuda()
	  canvas[t-1] = canvas[t-1]:cuda()
	  --ascending = ascending:cuda()

	  z[t], loss_z[t], lstm_c_enc[t], lstm_h_enc[t] = unpack(encoder_clones[t]:forward({x[t], x_error[t-1], lstm_c_enc[t-1], lstm_h_enc[t-1], e[t]}))
	  --[[print('z')
	  print(z[t]:size())
	  print('loss_z')
	  print(loss_z[t]:size())
	  print('lstm_c_enc')
	  print(lstm_c_enc[t]:size())
	  print('lstm_h_enc')
	  print(lstm_h_enc[t]:size())
	  print('patch')
	  print(patch[t]:size())
	  print('x')
	  print(x[t]:size())
	  print('encoder_clones')
	  print(encoder_clones[t]:size())
	  print('encoder_clones:x')
	  print(encoder_clones[t].x:size())]]--
	  
      x_prediction[t], x_error[t], lstm_c_dec[t], lstm_h_dec[t], canvas[t], loss_x[t] = unpack(decoder_clones[t]:forward({x[t], z[t], lstm_c_dec[t-1], lstm_h_dec[t-1], canvas[t-1]}))
      --print(patch[1]:gt(0.5))
      
      loss = loss + torch.mean(loss_z[t]) + torch.mean(loss_x[t])
    end
    loss = loss / seq_length
    print(loss)

    ------------------ backward pass -------------------
    -- complete reverse order of the above
    dlstm_c_enc = {[seq_length] = torch.zeros(n_data, rnn_size)}
    dlstm_h_enc = {[seq_length] = torch.zeros(n_data, rnn_size)}
    dlstm_c_dec = {[seq_length] = torch.zeros(n_data, rnn_size)}
    dlstm_h_dec = {[seq_length] = torch.zeros(n_data, rnn_size)}

    dx_error = {[seq_length] = torch.zeros(n_data, n_channels, A, B)}
    dx_prediction = {}
    dloss_z = {}
    dloss_x = {}
    dcanvas = {[seq_length] = torch.zeros(n_data, n_channels, A, B)}
    dz = {}
    dx1 = {}
    dx2 = {}
    de = {}
    --dpatch = {}
    
    for t = seq_length,1,-1 do
      dloss_x[t] = torch.ones(n_data, 1):cuda()
      dloss_z[t] = torch.ones(n_data, 1):cuda()
      dx_prediction[t] = torch.zeros(n_data, n_channels, 28, 28):cuda()
      --dpatch[t] = torch.zeros(n_data, N, N):cuda()

	  dx_error[t] = dx_error[t]:cuda()
	  dlstm_c_dec[t] = dlstm_c_dec[t]:cuda()
	  dlstm_h_dec[t] = dlstm_h_dec[t]:cuda()
	  dcanvas[t] = dcanvas[t]:cuda()
	  
  	  dlstm_c_enc[t] = dlstm_c_enc[t]:cuda()
  	  dlstm_h_enc[t] = dlstm_h_enc[t]:cuda()
  	  	  
      dx1[t], dz[t], dlstm_c_dec[t-1], dlstm_h_dec[t-1], dcanvas[t-1] = unpack(decoder_clones[t]:backward({x[t], z[t], lstm_c_dec[t-1], lstm_h_dec[t-1], canvas[t-1]}, {dx_prediction[t], dx_error[t], dlstm_c_dec[t], dlstm_h_dec[t], dcanvas[t], dloss_x[t]}))
      dx2[t], dx_error[t-1], dlstm_c_enc[t-1], dlstm_h_enc[t-1], de[t] = unpack(encoder_clones[t]:backward({x[t], x_error[t-1], lstm_c_enc[t-1], lstm_h_enc[t-1], e[t]}, {dz[t], dloss_z[t], dlstm_c_enc[t], dlstm_h_enc[t]}))

    end

    -- clip gradient element-wise
    grad_params:clamp(-5, 5)

    return loss, grad_params
end

------------------------------------------------------------------------
-- optimization loop
--

lr = 1e-2
optim_state = {learningRate = lr}

for i = 1, 100 do
    if i % 33 == 0 then
        lr = lr / 2
        optim_state = {learningRate = lr}
    end

    local _, loss = optim.adagrad(feval, params, optim_state)

    if i % 10 == 0 then
        print(string.format("iteration %4d, loss = %6.6f", i, loss[1]))
        --print(params)
    end
end

torch.save('x_prediction', x_prediction)

--generation
for t = 1, seq_length do
      e[t] = torch.randn(n_data, n_z):cuda()
      x[t] = features_input:cuda()
      z[t] = torch.randn(n_data, n_z):cuda()
      x_prediction[t], x_error[t], lstm_c_dec[t], lstm_h_dec[t], canvas[t], loss_x[t] = unpack(decoder_clones[t]:forward({x[t], z[t], lstm_c_dec[t-1], lstm_h_dec[t-1], canvas[t-1]}))
end

torch.save('x_generation', x_prediction)

