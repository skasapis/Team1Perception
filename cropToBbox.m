% use this file to crop all images in the trainval set to their ground
% truth bounding boxes ROI

trainds = imageDatastore('deploy/trainval/*/*_image.jpg');

numTrain = numel(trainds.Files);
[bbox, trainIdx] = BBox_Code(numTrain);
vehicle = bbox';% transposed so that each are nx1 shaped
imageFilename = trainds.Files(trainIdx);
trainingData = table(imageFilename, vehicle);

%% VALIDATE IMAGE/BOUNDING BOX DATA
% Vehicle data [x pos, y pos, xsize, ysize];

% Display first few rows of the data set.
% % trainingData(1:4,:)

for idx = 1:numTrain
    %Read one of the images.
    I = imread(trainingData.imageFilename{idx});

    % Insert the ROI labels.
    I = insertShape(I, 'Rectangle', trainingData.vehicle{idx});

    % Resize and display image.
    figure(1)
    imshow(I)

    cropI = imcrop(I, trainingData.vehicle{idx});
    figure(2)
    imshow(cropI)
    
    name = trainds.Files(idx);
    name = name{1}(end-50:end);
    folderName = name(1:end-15);
    
    cd deployCropped2/trainval
    status = mkdir(folderName);
    cd ../../
    
    filename = ['deployCropped2/trainval/', name]
    imwrite(cropI, filename)
end

% FOLDERS WITH VARIABLILITY IN CLASS
% 215
% fc26
% f075