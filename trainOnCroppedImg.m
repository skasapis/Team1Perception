%% transfer learning
% http://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html
% https://matlabacademy.mathworks.com


% add images to datastore
testds = imageDatastore('deploy/test/9b0e6947-340d-419d-ae8d-e73993afec6a/*_image.jpg'); %53 pictures
trainds = imageDatastore('deploy/test/047b864f-0753-448b-9483-f990ae41abaf/*_image.jpg'); %110 pictures

fname = trainds.Files; %names of all files
tname = trainds.Files;

numTrain = numel(fname);
numTest = numel(tname);

% Add labels to training data
groundTruth = csvread('deploy/trainval/labels.csv',2,2);
trainImgs.Labels = groundTruth(1:numTrain); %110


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
[carNet, info] = trainNetwork(trainds, layers, options);

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

numCorrect = nnz(flwrPreds == flwrActual)
fracCorrect = numCorrect/24
confusionchart(testImgs.Labels, flwrPreds)

