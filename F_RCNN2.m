%https://www.mathworks.com/help/vision/examples/object-detection-using-faster-r-cnn-deep-learning.html
%https://www.mathworks.com/help/vision/ref/trainfasterrcnnobjectdetector.html
%https://www.mathworks.com/help/vision/ref/trainfasterrcnnobjectdetector.html#bvkk009-1-trainingData

%% Load vehicle data set

%clc; clear all;
%data = load('fasterRCNNVehicleTrainingData.mat');
%dat = data.detector;
%lay = data.layers;
%clear data
bbds = dir('deploy/trainval/*/*_bbox.bin'); %run from inside the folder, easier to make consistent among all users

% imds = imageDatastore('deploy/trainval/*/*_image.jpg');
trainds = imageDatastore('deploy/trainval/*/*_image.jpg');
testds = imageDatastore('deploy/test/*/*_image.jpg');
numTrain = 5;
bbox = BBox_Code(numTrain);
% transposed so that each are nx1 shaped
vehicle = bbox';
imageFilename = imds.Files(1:numTrain);
trainingData = table(imageFilename, vehicle);

% vehicleDataset.Var1 is their equivilent of vehicleDataset.imageFilename
% vehicleDataset.Var2 is their equivilent of vehicleDataset.vehicle


%% VALIDATE IMAGE/BOUNDING BOX DATA
% Vehicle data [x pos, y pos, xsize, ysize];

% Display first few rows of the data set.
% % % trainingData(1:4,:)

% Read one of the images.
% % % I = imread(trainingData.imageFilename{5});
% % % 
% % % % Insert the ROI labels.
% % % I = insertShape(I, 'Rectangle', trainingData.vehicle{5});
% % % 
% % % % Resize and display image.
% % % I = imresize(I,3);
% % % figure(1)
% % % imshow(I)
%% Split data into a training and test set.
% idx = floor(0.6 * height(vehicleDataset));
% trainingData = vehicleDataset(1:idx,:);
% testData = vehicleDataset(idx:end,:);
%traininData = vehicleDataset;

%% BUILD NETWORK
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
    fullyConnectedLayer(width(trainingData))

    % Add the softmax loss layer and classification layer. 
    softmaxLayer()
    classificationLayer()
];

layers = [
    inputLayer
    middleLayers
    finalLayers
    ]

% Options for step 1.
optionsStage1 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 1, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir);

% Options for step 2.
optionsStage2 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 1, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir);

% Options for step 3.
optionsStage3 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 1, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir);

% Options for step 4.
optionsStage4 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 1, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir);

options = [
    optionsStage1
    optionsStage2
    optionsStage3
    optionsStage4
    ];

%% TRAIN NETWORK
% A trained network is loaded from disk to save time when running the
% example. Set this flag to true to train the network. 
doTrainingAndEval = true;

if doTrainingAndEval
    % Set random seed to ensure example training reproducibility.
    rng(0);
    
    % Train Faster R-CNN detector. Select a BoxPyramidScale of 1.2 to allow
    % for finer resolution for multiscale object detection.
    detector = trainFasterRCNNObjectDetector(trainingData, layers, options, ...
        'NegativeOverlapRange', [0 0.3], ...
        'PositiveOverlapRange', [0.6 1], ...
        'NumRegionsToSample', [256 128 256 128], ...
        'BoxPyramidScale', 1.2);
else
    % Load pretrained detector for the example.
    detector = data.detector;
end

disp('NETWORK TRAINED');



%% TEST TRAINED NETWORK
% Read a test image.
I = imread(testData.Files{41});

% Run the detector.
[bboxes,scores] = detect(detector,I);

% Annotate detections in the image.
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
% save image
imwrite(I, 'detectTest.png')

disp('SINGLE TEST IMAGE DETECTION COMPLETE')



%% RUN DETECTOR ON ALL TEST IMAGES
detectAll = 0;
if detectAll == 1
    % Annotate detections in the image.
    I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
    % % % figure
    % % % imshow(I)

    if doTrainingAndEval
        % Run detector on each image in the test set and collect results.
        resultsStruct = struct([]);
        for i = 1:height(testData)

            % Read the image.
            I = imread(testData.Files{i});

            % Run the detector.
            [bboxes, scores, labels] = detect(detector, I);

            % Collect the results.
            resultsStruct(i).Boxes = bboxes;
            resultsStruct(i).Scores = scores;
            resultsStruct(i).Labels = labels;
        end

        % Convert the results into a table.
        results = struct2table(resultsStruct);
    else
        % Load results from disk.
        results = data.results;
    end

    %% Extract expected bounding box locations from test data.
    expectedResults = testData(:, 2:end);

    % Evaluate the object detector using Average Precision metric.
    [ap, recall, precision] = evaluateDetectionPrecision(results, expectedResults);

    % Plot precision/recall curve
    figure(2)
    plot(recall,precision)
    xlabel('Recall')
    ylabel('Precision')
    grid on
    title(sprintf('Average Precision = %.2f', ap))
    print('Precision', '-dpng')

    disp('TEST IMAGE DETECTION COMPLETE')
    
end % end of detectAll == true






%% ///////////////////// SUPPLEMENTARY FUNCTIONS /////////////////////
function data = read_bin(file_name)
    id = fopen(file_name, 'r');
    data = fread(id, inf, 'single');
    fclose(id);
end
