

Tensor = torch.CudaTensor

-- number of classes and examples per class in a batch for training
n_classes = 15
n_samples = 20

-- Network hyperparameters
text_feat_size = 1024
n_z = 100			--20    --400
rnn_size = 256		--100   --1024
seq_length = 5		--50
-- input image channels
n_channels = 3
--N = 15				--3
-- Image Height
A = 32
-- Image Width
B = 32
n_data = n_classes*n_samples
n_canvas = A*B

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
    local file_rand = torch.randperm(150)--#file_list)

    image_features = torch.zeros(n_data, n_channels, A, B)
    text_features = torch.zeros(n_data, 10, text_feat_size)

    for i = 1, n_classes do
        -- Read image and text data for all samples in the class
        --print(file_list[file_rand[i]])
        full_image_data = torch.load(file_list[file_rand[i]])
        text_file = string.match(file_list[file_rand[i]], '%d%d%d%..*%.t7')
        full_text_data = torch.load(paths.concat(text_dir, text_file))

        -- Read n_samples random samples from each class
        sample_rand = torch.randperm(full_image_data:size(1))
        for j = 1, n_samples do
            image_features[{{(i-1)*n_samples+j}, {}, {}, {}}] = full_image_data[sample_rand[j]]
            for k = 1, 10 do
                text_features[{{(i-1)*n_samples+j}, {k}, {}}] = full_text_data['txt_fea'][(sample_rand[j]-1)*10 + k]
            end
        end
    end
    print(image_features:size())
    print(text_features:size())
    return image_features, text_features
end

