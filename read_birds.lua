--
-- Created by IntelliJ IDEA.
-- User: aabdujyo
-- Date: 23/02/17
-- Time: 3:46 PM
-- To change this template use File | Settings | File Templates.
--
require 'torch'
require 'image'
require 'image'
require 'cutorch'

local model_utils=require 'model_utils'

files = {}

--dir_name = "/cs/vml4/Datasets/Caltech-UCSD-Birds-200/images"
dir_name = "/cs/vml4/zhiweid/ICML17/Reed_cvpr16/txt_im_pair"
file_extention = ".t7"

-- 2. Load all files in directory

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


-- 3. Sort file names

table.sort(files, function (a,b) return a < b end)

print('Found files:')
print(files)


-- 4. Finally we load images

images = {}
for i,file in ipairs(files) do
   -- load each image
   table.insert(images, torch.load(file))
end
print(#image)

print('Loaded images:')
print(images)

-- Display a of few them
--[[for i = 1,math.min(#files,10) do
   image.display{image=images[i], legend=files[i]}
end]]--
