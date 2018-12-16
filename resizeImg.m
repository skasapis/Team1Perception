% to resize images to be the correct input size to googlenet
trainds = imageDatastore('deploy/trainval/*/*_image.jpg');
for idx = 1:10
    
    name = trainds.Files{idx};
    I = imread(trainds.Files{idx});
    J = imresize(I,[224 224]);
    
    name = name(end-50:end);
    filename = ['deploy/trainvalGnetSize/', name]
    imwrite(J, filename)
end