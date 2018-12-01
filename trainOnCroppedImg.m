%% transfer learning
% http://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html
% https://matlabacademy.mathworks.com

% add images to datastore
    % I had to pick one that started with a letter to the because
    % matlab doesn't like names that start with a number
testds = imageDatastore('deploy/test/d7f7c063-df18-4f96-a071-aa634b9e502e/*_image.jpg'); %53 pictures
trainds = imageDatastore('deploy/trainval/af3ae5e6-27ef-4699-beb3-f2c72831a594/*_image.jpg'); %223 pictures

fname = trainds.Files; %names of all files
tname = testds.Files;

numTrain = numel(fname);
numTest = numel(tname);

% Add labels to training data
groundTruth = csvread('deploy/trainval/labels.csv',1,1);
trainds.Labels = groundTruth(1:numTrain); %110


f1 = fname{1}



%Create a network by modifying AlexNet
numClasses = 3;
net = alexnet;
layers = net.Layers;
fc = fullyConnectedLayer(numClasses); %create a new fully connected layer with n nodes
layers(end-2) = fc; % replace layer with new fc layer
layers(end) = classificationLayer;

% set initial training weights with an initial conservative learning rate
% (weight changing step) 
    % ?stochastic gradient descent with momentum?.
options = trainingOptions('sgdm', 'InitialLearnRate', 0.001);

%Perform training
tic
[carNet, info] = trainNetwork(trainds, layers, options);
toc

%Use trained network to classify single test image
img = readimage(testds,1);
[testpreds, scores] = classify(carNet, img);


% display bar graph of scores in classification
figure(1)
outlayer = layers(end);
categorynames = outlayer.Classes;
thresh = median(scores) + std(scores); % based on being more than a std away
highscores = scores > thresh;
bar(scores(highscores))
% xticks(1:length(scores(highscores)))
xticklabels(categorynames(highscores))
xtickangle(60)


%
figure(2)
plot(info.TrainingLoss)

%
figure(3)
confusionchart(testImgs.Labels, flwrPreds)

% print statistics
numCorrect = nnz(flwrPreds == flwrActual)
fracCorrect = numCorrect/24


