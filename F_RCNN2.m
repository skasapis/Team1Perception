% https://www.mathworks.com/help/vision/examples/object-detection-using-faster-r-cnn-deep-learning.html
% https://www.mathworks.com/help/vision/ref/trainfasterrcnnobjectdetector.html
% https://www.mathworks.com/help/vision/ref/trainfasterrcnnobjectdetector.html#bvkk009-1-trainingData
% https://www.mathworks.com/help/vision/ref/fasterrcnnobjectdetector.detect.html
% https://www.mathworks.com/help/vision/ug/faster-r-cnn-basics.html

%% Load vehicle data set

trainds = imageDatastore('deploy/trainval/*/*_image.jpg');
testds = imageDatastore('deploy/test/*/*_image.jpg');

%% VALIDATE IMAGE/BOUNDING BOX DATA
% Vehicle data [x pos, y pos, xsize, ysize];

% Display first few rows of the data set.
% % % trainingData(1:4,:)

% Read one of the images.
% % % I = imread(trainingData.imageFilename{5});
% % % 
% % % % Insert the ROI labels.
% % % box = trainingData.vehicle{5};
% % % box2 = [box(1:2), 2, 2];
% % % I2 = insertShape(I, 'Rectangle',box); % whole box
% % % I2 = insertShape(I, 'Rectangle', box2,'Color', {'green'}); % upper left corner
% % % 
% % % % Resize and display image.
% % % figure(1)
% % % imshow(I2)
% % % 
% % % cropI = imcrop(I, box);
% % % figure(2)
% % % imshow(cropI)


%% TRAIN DETECTOR
% A trained network is loaded from disk to save time when running the
% example. Set this flag to true to train the network. 
doTrainingAndEval = false;
loadPrev = true;

tic
if doTrainingAndEval
    
    % create data table to feed into train RCNN
    numTrain = 38;
    [bbox, trainIdx] = BBox_Code(numTrain);
    vehicle = bbox';% transposed so that each are nx1 shaped
    imageFilename = trainds.Files(trainIdx);
    trainingData = table(imageFilename, vehicle);
    
    % Set random seed to ensure example training reproducibility.
    rng(0);
     
    % Train Faster R-CNN detector. Select a BoxPyramidScale of 1.2 to allow
    % for finer resolution for multiscale object detection.
    
    % OPTION 1:
%     numTrain = width(trainingData);
%     [layers, options] = buildRCNN(numTrain) % in function at bottom to clean code
    
    % OPTION 2:
%     anet = alexnet;
%     layers = anet.Layers;
    
    % OPTION 3:
    data = load('fasterRCNNVehicleTrainingData.mat');
    detector = data.detector;
    numTrain = width(trainingData);
    [~, options] = buildRCNN(numTrain);
    
    cdetector = trainFasterRCNNObjectDetector(trainingData, detector, options, ...
        'SmallestImageDimension', 500, ...
        'NegativeOverlapRange', [0 0.3], ...
        'PositiveOverlapRange', [0.6 1], ...
        'BoxPyramidScale', 1.2);%, ...
%         'NumRegionsToSample', [256 128 256 128]);
    disp('DETECTOR TRAINED');
    save cdetector

elseif loadPrev
    load cdetector
    disp('DETECTOR LOADED');
else
    % Load pretrained detector from the example.
    data = load('fasterRCNNVehicleTrainingData.mat');
    cdetector = data.detector;
    disp('EXAMPLE DETECTOR LOADED');
end
toc


%% APPLY DETECTOR TO TEST IMAGES

tic
for idx = [41 107 108 116 117 118 119 500 528 1604 1964]%1:height(testData) 

    % Read the image.
    I = imread(testds.Files{idx});

    % Run the detector.
    [detbbox, scores, labels] = detect(cdetector,I)

    % crop image and save
    if numel(detbbox < 4) == 0
        % save original image -- no cropping
        cropI = I;
%         imshow(cropI)
    else       
        [mx, maxScoreIdx] = max(scores);
        % draw box and save image
        I2 = insertShape(I, 'Rectangle', detbbox(maxScoreIdx,1:4));
        imwrite(I2, 'detectTest.png');
%         figure(1)
%         imshow(I2)
        
        cropI = imcrop(I, detbbox(maxScoreIdx,1:4));
        imwrite(cropI, 'detectCrop.png');
%         figure(2)
%         imshow(cropI)
    end

    name = testds.Files(idx);
    name = name{1}(end-50:end);
    filename = ['deployCropped/', name]
    imwrite(cropI, filename)

end

disp('TEST IMAGE DETECTION COMPLETE')
toc





%% ///////////////////// SUPPLEMENTARY FUNCTIONS /////////////////////

function data = read_bin(file_name)
    id = fopen(file_name, 'r');
    data = fread(id, inf, 'single');
    fclose(id);
end


function [layers, options] = buildRCNN(numTrain)

    % Create image input layer.
    inputLayer = imageInputLayer([32 32 3]);

    % Define the convolutional layer parameters.
    filterSize = [3 3];
    numFilters = 32;

    % Create the middle layers.
    middleLayers = [            
        convolution2dLayer(filterSize, numFilters, 'Padding', 1)   
        reluLayer()
        convolution2dLayer(filterSize, numFilters, 'Padding', 1)  
        reluLayer() 
        maxPooling2dLayer(3, 'Stride',2)    
        ];

    finalLayers = [
        % Add a fully connected layer with 64 output neurons. The output size
        % of this layer will be an array with a length of 64.
        fullyConnectedLayer(64)
        % Add a ReLU non-linearity.
        reluLayer()
        % Add the last fully connected layer. At this point, the network must
        % produce outputs that can be used to measure whether the input image
        % belongs to one of the object classes or background. This measurement
        % is made using the subsequent loss layers.
        fullyConnectedLayer(numTrain)
        % Add the softmax loss layer and classification layer. 
        softmaxLayer()
        classificationLayer()
    ];

    layers = [
        inputLayer
        middleLayers
        finalLayers
        ];

    % use googlenet for transfer learning rather than train from scratch
    % gnet=googlenet;
    % lgraph = layerGraph(net);

    batchSz = 5
    
    % Options for step 1.
    optionsStage1 = trainingOptions('sgdm', ...
        'MaxEpochs', 4, ...
        'MiniBatchSize', batchSz, ...
        'InitialLearnRate', 1e-3, ...
        'CheckpointPath', tempdir, ...
        'VerboseFrequency', 10);

    % Options for step 2.
    optionsStage2 = trainingOptions('sgdm', ...
        'MaxEpochs', 5, ...
        'MiniBatchSize', batchSz, ...
        'InitialLearnRate', 1e-3, ...
        'CheckpointPath', tempdir, ...
        'VerboseFrequency', 10);

    % Options for step 3.
    optionsStage3 = trainingOptions('sgdm', ...
        'MaxEpochs', 5, ...
        'MiniBatchSize', batchSz, ...
        'InitialLearnRate', 1e-3, ...
        'CheckpointPath', tempdir, ...
        'VerboseFrequency', 10);

    % Options for step 4.
    optionsStage4 = trainingOptions('sgdm', ...
        'MaxEpochs', 5, ...
        'MiniBatchSize', batchSz, ...
        'InitialLearnRate', 1e-3, ...
        'CheckpointPath', tempdir, ...
        'VerboseFrequency', 10);

    options = [
        optionsStage1
        optionsStage2
        optionsStage3
        optionsStage4
        ];


end

