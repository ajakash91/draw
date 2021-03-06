--require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'
local model_utils=require 'model_utils'
--local mnist = require 'mnist'
require 'cutorch'
require 'cunn'
require 'GaussianCriterion'
--dbg = require 'debugger.lua'

--nngraph.setDebug(true)

--Tensor = torch.CudaTensor

-- number of classes and examples per class in a batch for training
n_classes = 1
n_samples = 20

-- Network hyperparameters
text_feat_size = 1024
n_z = 100			--20    --400
rnn_size = 256		--100   --1024
seq_length = 5   	--50
-- input image channels
n_channels = 3
--N = 15				--3
-- Image Height
A = 32
-- Image Width
B = 32
n_data = n_classes*n_samples
n_canvas = A*B
normalization = 'batch-norm' -- 'batch-norm', 'weight-norm', or 'none'

o1 = 12
--o2 = 32
--o3 = 32
f1 = 5
--f2 = 3
--f3 = 3
final_width = A-f1+1

text_dir = '/cs/vml4/zhiweid/ICML17/Reed_cvpr16/txt_im_pair/'

-- Create a list of image directories
image_dir = '/cs/vml4/Datasets/Caltech-UCSD-Birds-200/images_32/'
file_list = {}
for file in paths.files(image_dir) do
    if string.match(file, '%d%d%d%..*%.t7') then
        table.insert(file_list, paths.concat(image_dir, file))
    end
end
-- Check files and sort them
if #file_list == 0 then
    error('given directory doesnt contain any files of type: ' .. opt.ext)
end
table.sort(file_list, function (a,b) return a < b end)

-- Select n_classes random classes and n_sample random images from each of those classes
function read_data()
    -- List of random numbers to select from test set class (150 classes)
    local file_rand = torch.randperm(n_classes)--(150)--#file_list)

    image_features = torch.zeros(n_data, n_channels, A, B)
    --text_features = torch.zeros(n_data, 10, text_feat_size)

    for i = 1, n_classes do
        -- Read image and text data for all samples in the class
        --print(file_list[file_rand[i]])
        full_image_data = torch.load(file_list[file_rand[i]])
        --text_file = string.match(file_list[file_rand[i]], '%d%d%d%..*%.t7')
        --full_text_data = torch.load(paths.concat(text_dir, text_file))

        -- Read n_samples random samples from each class
        sample_rand = torch.randperm(n_samples)--full_image_data:size(1))
        --sample_rand = torch.range(n_samples)--full_image_data:size(1))
        for j = 1, n_samples do
            image_features[{{(i-1)*n_samples+j}, {}, {}, {}}] = full_image_data[j]--sample_rand[j]]
            --[[for k = 1, 10 do
                text_features[{{(i-1)*n_samples+j}, {k}, {}}] = full_text_data['txt_fea'][(sample_rand[j]-1)*10 + k]
            end]]--
        end
    end
    --print(image_features:size())
    --print(text_features:size())
    return image_features
end

function LSTM_old(inp, pr_h, pr_c, n_in, n_rnn)
    local function new_input_sum()
    -- transforms input
    --local i2h            = nn.BatchNormalization(n_rnn)(nn.Linear(n_in, n_rnn)(inp))
    local i2h            = nn.Linear(n_in, n_rnn)(inp)
    -- transforms previous timestep's output
    --local h2h            = nn.BatchNormalization(n_rnn)(nn.Linear(n_rnn, n_rnn)(pr_h))
    local h2h            = nn.Linear(n_rnn, n_rnn)(pr_h)
    return nn.CAddTable()({i2h, h2h})
    end

    local in_gate          = nn.Sigmoid()(new_input_sum())
    local forget_gate      = nn.Sigmoid()(new_input_sum())
    local out_gate         = nn.Sigmoid()(new_input_sum())
    local in_transform     = nn.Tanh()(new_input_sum())

    nx_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, pr_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    nx_h           = nn.CMulTable()({out_gate, nn.Tanh()(nx_c)})

    return nx_c, nx_h
end

function enc_convolution(x)
    if normalization == 'batch-norm' then
        layer1 = nn.ReLU()(nn.SpatialBatchNormalization(o1)(nn.SpatialConvolution(n_channels, o1, f1, f1)(x)))
    elseif normalization == 'weight-norm' then
        --local weight_norm =
    else
        layer1 = nn.ReLU()(nn.SpatialConvolution(n_channels, o1, f1, f1)(x))
    end
	--layer2 = nn.ReLU()(nn.SpatialConvolution(o1, o2, f2, f2)(layer1))
	--layer3 = nn.ReLU()(nn.SpatialConvolution(o2, o3, f3, f3)(layer2))
	layer3_flat = nn.View(o1*(final_width)*(final_width))(layer1)
	fc = nn.ReLU()(nn.Linear(o1*(final_width)*(final_width), rnn_size)(layer3_flat))

    return(fc)
end

function dec_convolution(next_h)
    if normalization == 'batch-norm' then
        fc_1 = nn.BatchNormalization(o1*(final_width)*(final_width))(nn.Linear(rnn_size, o1*(final_width)*(final_width))(next_h))
	elseif normalization == 'weight-norm' then
        --a=decoder
    else
        fc_1 = nn.Linear(rnn_size, o1*(final_width)*(final_width))(next_h)
    end

    fc_1 = nn.View(o1, (final_width), (final_width))(fc_1)
	fc_1 = nn.ReLU()(fc_1)
	--layer_1 = nn.ReLU()(nn.SpatialFullConvolution(o3, o2, f3, f3)(fc_1))
	--layer_2 = nn.ReLU()(nn.SpatialFullConvolution(o2, o1, f2, f2)(fc_1))
	layer_3 = nn.Sigmoid()(nn.SpatialFullConvolution(o1, n_channels, f1, f1)(fc_1))

    return(layer_3)
end

function enc_linear(x)
    x = nn.View(n_channels*A*B)(x)
	layer1 = nn.ReLU()(nn.Linear(n_channels*A*B, 500)(x))
	fc = nn.ReLU()(nn.Linear(500, rnn_size)(layer1))

    return(fc)
end

function dec_linear(next_h)
	fc_1 = nn.ReLU()(nn.Linear(rnn_size, 500)(next_h))
	layer_3 = nn.Linear(500, n_channels*A*B)(fc_1)
    layer_3 = nn.Sigmoid()(nn.View(n_channels, A, B)(layer_3))

    return(layer_3)
end

--encoder
x = nn.Identity()()
x_error_prev = nn.Identity()()

fc1 = enc_convolution(x)
fc1_e = enc_convolution(x_error_prev)
--fc1 = enc_linear(x)
--fc1_e = enc_linear(x_error_prev)

input = nn.JoinTable(1)({fc1, fc1_e})
input = nn.View(rnn_size*2)(input)
n_input = rnn_size*2

prev_h = nn.Identity()()
prev_c = nn.Identity()()

next_c, next_h = LSTM_old(input, prev_h, prev_c, n_input, rnn_size)

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
encoder = nn.gModule({x, x_error_prev, prev_c, prev_h, e}, {z, loss_z, next_c, next_h})
encoder = encoder:cuda()
encoder.name = 'encoder'

--decoder
x = nn.Identity()()
z = nn.Identity()()
prev_h = nn.Identity()()
prev_c = nn.Identity()()
--prev_canvas = nn.Identity()()
n_input = n_z
input = z

next_c, next_h = LSTM_old(input, prev_h, prev_c, n_input, rnn_size)

-- write layer
mu_prediction = dec_convolution(next_h)
sigma_prediction = dec_convolution(next_h)
--mu_prediction = dec_linear(next_h)
--sigma_prediction = dec_linear(next_h)

--write layer end

--next_canvas = nn.CAddTable()({prev_canvas, mu_prediction})

--mu = nn.Sigmoid()(next_canvas)

neg_mu = nn.MulConstant(-1)(mu_prediction)
d = nn.CAddTable()({x, neg_mu})

mu_prediction = nn.View(n_channels, A, B)(mu_prediction)
sigma_prediction = nn.View(n_channels, A, B)(sigma_prediction)

x_error = nn.View(n_channels, A, B)(d)

decoder = nn.gModule({x, z, prev_c, prev_h}, {mu_prediction, sigma_prediction, x_error, next_c, next_h})
decoder = decoder:cuda()
decoder.name = 'decoder'

params, grad_params = model_utils.combine_all_parameters(encoder, decoder)

encoder_clones = model_utils.clone_many_times(encoder, seq_length)
decoder_clones = model_utils.clone_many_times(decoder, seq_length)

criterion = nn.GaussianCriterion()

print_count = 0
-- do fwd/bwd and return loss, grad_params
function feval(x_arg)
    -- Read images
    img_features_input = {}
    img_features_input = read_data()

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
    mu_prediction = {}
    sigma_prediction = {}
    loss_z = {}
    loss_x = {}
    dx_prediction = {}
    dmu_prediction = {}
    dsigma_prediction = {}

    --canvas = {[0]=torch.rand(n_data, n_channels, A, B)}
    x = {}
    --patch = {}

    local loss = 0
    local losss_z = 0
    local losss_x = 0

    for t = 1, seq_length do
        e[t] = torch.randn(n_data, n_z):cuda()
        x[t] = img_features_input:cuda()

        lstm_h_enc[t-1] = lstm_h_enc[t-1]:cuda()
        lstm_c_enc[t-1] = lstm_c_enc[t-1]:cuda()
        lstm_h_dec[t-1] = lstm_h_dec[t-1]:cuda()
        lstm_c_dec[t-1] = lstm_c_dec[t-1]:cuda()
        x_error[t-1] = x_error[t-1]:cuda()
        --canvas[t-1] = canvas[t-1]:cuda()
        --ascending = ascending:cuda()

        z[t], loss_z[t], lstm_c_enc[t], lstm_h_enc[t] = unpack(encoder_clones[t]:forward({x[t], x_error[t-1], lstm_c_enc[t-1], lstm_h_enc[t-1], e[t]}))

        mu_prediction[t], sigma_prediction[t], x_error[t], lstm_c_dec[t], lstm_h_dec[t]= unpack(decoder_clones[t]:forward({x[t], z[t], lstm_c_dec[t-1], lstm_h_dec[t-1]}))

        --ln_sigma_prediction[t] = math.log(sigma_prediction[t]:pow(2))

        x_prediction[t] = {mu_prediction[t], sigma_prediction[t] }

        loss_x[t] = criterion:forward(x_prediction[t], x[t])
        dx_prediction[t] = criterion:backward(x_prediction[t], x[t])

        dmu_prediction[t] = dx_prediction[t][1]
        dsigma_prediction[t] = dx_prediction[t][2]

        --[[print('z')
        print(z[t]:size())
        print('loss_z')
        print(loss_z[t]:size())
        print('lstm_c_enc')
        print(lstm_c_enc[t]:size())
        print('lstm_h_enc')
        print(lstm_h_enc[t]:size())
        print('loss_z')
        print(loss_z[t]:size())
        print('x')
        print(x[t]:size())
        print('x_error')
        print(x_error[t]:size())
        print('e')
        print(e[t]:size())
        print('mu_prediction')
        print(mu_prediction[t]:size())
        print('sigma_prediction')
        print(sigma_prediction[t]:size())
        print('lstm_c_dec')
        print(lstm_c_dec[t]:size())
        print('x_prediction')
        print(#x_prediction[t])
        print('dx_prediction')
        print(#dx_prediction[t])
        print('dmu_prediction')
        print(dmu_prediction[t]:size())
        print('loss_x')
        print(loss_x[t])]]--

        --loss = loss + torch.mean(loss_z[t]) + (loss_x[t] / (n_channels * A * B)) -- torch.mean(loss_x[t])
        losss_z = losss_z + torch.mean(loss_z[t])
        losss_x = losss_x + loss_x[t] / (n_channels * A * B)
    end
    losss_x = losss_x / seq_length
    losss_z = losss_z / seq_length

    loss = losss_x + losss_z
    --print(losss_x, losss_z)
    --print(mu_prediction[1]:size())
    print_count = print_count + 1
    if print_count % 100 == 0 then
        start = torch.random(1, 140)
        print(losss_x, losss_z)
        for ind = 1, n_classes*n_samples do
            image.save('tmp/1/sample_reconstruction_'..ind ..'.jpg', mu_prediction[seq_length][ind])
            image.save('tmp/1/sample_'..ind ..'.jpg', x[seq_length][ind])
        end
    end

    ------------------ backward pass -------------------
    -- complete reverse order of the above
    dlstm_c_enc = {[seq_length] = torch.zeros(n_data, rnn_size)}
    dlstm_h_enc = {[seq_length] = torch.zeros(n_data, rnn_size)}
    dlstm_c_dec = {[seq_length] = torch.zeros(n_data, rnn_size)}
    dlstm_h_dec = {[seq_length] = torch.zeros(n_data, rnn_size)}

    dx_error = {[seq_length] = torch.zeros(n_data, n_channels, A, B)}
    --dx_prediction = {}
    dloss_z = {}
    --dcanvas = {[seq_length] = torch.zeros(n_data, n_channels, A, B)}
    dz = {}
    dx1 = {}
    dx2 = {}
    de = {}
    --dpatch = {}

    for t = seq_length,1,-1 do
        --dloss_x[t] = torch.ones(n_data, 1):cuda()
        dloss_z[t] = torch.ones(n_data, 1):cuda()
        --dx_prediction[t] = torch.zeros(n_data, n_channels, A, B):cuda()
        --dpatch[t] = torch.zeros(n_data, N, N):cuda()

        dx_error[t] = dx_error[t]:cuda()
        dlstm_c_dec[t] = dlstm_c_dec[t]:cuda()
        dlstm_h_dec[t] = dlstm_h_dec[t]:cuda()
        --dcanvas[t] = dcanvas[t]:cuda()

        dlstm_c_enc[t] = dlstm_c_enc[t]:cuda()
        dlstm_h_enc[t] = dlstm_h_enc[t]:cuda()

        dx1[t], dz[t], dlstm_c_dec[t-1], dlstm_h_dec[t-1]= unpack(decoder_clones[t]:backward({x[t], z[t], lstm_c_dec[t-1], lstm_h_dec[t-1]}, {dmu_prediction[t], dsigma_prediction[t], dx_error[t], dlstm_c_dec[t], dlstm_h_dec[t]}))
        dx2[t], dx_error[t-1], dlstm_c_enc[t-1], dlstm_h_enc[t-1], de[t] = unpack(encoder_clones[t]:backward({x[t], x_error[t-1], lstm_c_enc[t-1], lstm_h_enc[t-1], e[t]}, {dz[t], dloss_z[t], dlstm_c_enc[t], dlstm_h_enc[t]}))
    end

    -- clip gradient element-wise
    grad_params:clamp(-5, 5)

    return loss, grad_params
end

------------------------------------------------------------------------
-- optimization loop
--

lr = 5e-4
optim_state = {learningRate = lr}

train_losses = {}
val_losses = {}

for i = 1, 300000 do
    if i % 5000 == 0 then
        lr = lr / 2
        optim_state = {learningRate = lr}
    end

    local _, loss = optim.adam(feval, params, optim_state)
    local train_loss = loss[1]
    train_losses[i] = train_loss

    if i % 10 == 0 then
        print(string.format("iteration %4d, loss = %6.6f", i, loss[1]))
        --print(params)
    end

    --[[if i % 25000 == 0 or i == 300000 then
        -- evaluate loss on validation data
        local val_loss = 0
        val_losses[i] = val_loss

        local savefile = string.format('output/chk_%d_%.5f_.t7', i, lr)
        print('saving checkpoint to output/' .. savefile)

        local checkpoint = {}
        checkpoint.encoder = encoder
        checkpoint.decoder = decoder
        checkpoint.train_losses = train_losses
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        torch.save(savefile, checkpoint)
    end]]--
end

torch.save('x_prediction', x_prediction)

--generation
for t = 1, seq_length do
    e[t] = torch.randn(n_data, n_z):cuda()
    x[t] = img_features_input:cuda()
    z[t] = torch.randn(n_data, n_z):cuda()
    mu_prediction[t], sigma_prediction[t], x_error[t], lstm_c_dec[t], lstm_h_dec[t], canvas[t], loss_x[t] = unpack(decoder_clones[t]:forward({x[t], z[t], lstm_c_dec[t-1], lstm_h_dec[t-1], canvas[t-1]}))
    x_prediction[t] = {mu_prediction[t], sigma_prediction[t]}
end

torch.save('x_generation', x_prediction)
