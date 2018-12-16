%https://www.mathworks.com/help/vision/examples/object-detection-using-faster-r-cnn-deep-learning.html
%https://www.mathworks.com/help/vision/ref/trainfasterrcnnobjectdetector.html
%https://www.mathworks.com/help/vision/ref/trainfasterrcnnobjectdetector.html#bvkk009-1-trainingData

%% Load vehicle data set

trainds = imageDatastore('deploy/trainval/*/*_image.jpg');
testds = imageDatastore('deploy/test/*/*_image.jpg');
numTrain = 500;
bbox = BBox_Code(numTrain);
% transposed so that each are nx1 shaped
vehicle = bbox';
imageFilename = trainds.Files(1:numTrain);
trainingData = table(imageFilename, vehicle);


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
    'MaxEpochs', 5, ...
    'MiniBatchSize', 10, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency', 200);

% Options for step 2.
optionsStage2 = trainingOptions('sgdm', ...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 10, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency', 200);

% Options for step 3.
optionsStage3 = trainingOptions('sgdm', ...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 10, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency', 200);

% Options for step 4.
optionsStage4 = trainingOptions('sgdm', ...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 10, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency', 200);

options = [
    optionsStage1
    optionsStage2
    optionsStage3
    optionsStage4
    ];

%% TRAIN DETECTOR
% A trained network is loaded from disk to save time when running the
% example. Set this flag to true to train the network. 
doTrainingAndEval = false;
loadPrev = true;

tic
if doTrainingAndEval
    % Set random seed to ensure example training reproducibility.
    rng(0);
    
    % Train Faster R-CNN detector. Select a BoxPyramidScale of 1.2 to allow
    % for finer resolution for multiscale object detection.
    detector = trainFasterRCNNObjectDetector(trainingData, layers, optionsStage1);%, ...
%         'NegativeOverlapRange', [0 0.3], ...
%         'PositiveOverlapRange', [0.6 1], ...
%         'BoxPyramidScale', 1.2);
%         'NumRegionsToSample', [256 128 256 128], ...
    disp('DETECTOR TRAINED');
    save detector

elseif loadPrev
    load detector
    disp('DETECTOR LOADED');
else
    % Load pretrained detector for the example.
    data = load('fasterRCNNVehicleTrainingData.mat');
    detector = data.detector;
    disp('BUILT IN DETECTOR LOADED');
end
toc




%% RUN DETECTOR
detectAll = false;
tic
if detectAll
    %% TEST TRAINED NETWORK ON ALL TEST IMAGES
    % Annotate detections in the image.
    % % %     I = insertObjectAnnotation(I,'rectangle',detbboxes,scores);
    % % % figure
    % % % imshow(I)


    % Run detector on each image in the test set and collect results.
    resultsStruct = struct([]);
    for idx = 1:height(testData)

        % Read the image.
        I = imread(testData.Files{idx});

        % Run the detector.
        [detbboxes, scores, labels] = detect(detector, I);

        % Collect the results.
        resultsStruct(idx).Boxes = detbboxes;
        resultsStruct(idx).Scores = scores;
        resultsStruct(idx).Labels = labels;
    end

    % Convert the results into a table.
    results = struct2table(resultsStruct);
    disp('TEST IMAGE DETECTION COMPLETE')
    
    
else % detectAll == false
    %% TEST TRAINED NETWORK ON SINGLE IMAGE
    % Read a test image.
    idx = 41;
    I = imread(testds.Files{idx});

    % Run the detector.
    [detbboxes,scores] = detect(detector,I);

    % Annotate detections in the image.
    %I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
    I = insertShape(I, 'Rectangle', detbboxes);
    % save image
    imwrite(I, 'detectTest.png')

    disp('SINGLE TEST IMAGE DETECTION COMPLETE')
    
    % CROP IMAGE AND SAVE TO NEW FOLDER
    xL = floor(detbboxes(1));
    xR = floor(detbboxes(1)+detbboxes(3));
    yT = floor(detbboxes(2));
    yB = floor(detbboxes(2)+detbboxes(4));
    cropI = I(yT:yB, xL:xR, 1:3);
    
    name = testds.Files(idx);
    name = name{1}(end-50:end);
    filename = ['deployCropped/', name];
    imwrite(cropI, filename)
    imwrite(cropI, 'detectCrop.png')
    
end
toc





%% ///////////////////// SUPPLEMENTARY FUNCTIONS /////////////////////
function data = read_bin(file_name)
    id = fopen(file_name, 'r');
    data = fread(id, inf, 'single');
    fclose(id);
end
