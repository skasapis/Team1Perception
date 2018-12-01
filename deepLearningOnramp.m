
%% load images into datastore and get path names
imds = imageDatastore('deploy/test/*/*_image.jpg');
fname = imds.Files %names of all files
f1 = fname{1}

% adding labels to dataset folders based on the folder they are in -- not
% helpful probably
flwrds = imageDatastore(pathToImages,'IncludeSubfolders',true,'LabelSource','foldernames');
flowernames = flwrds.Labels


%%

for n = 1:numel(fname)
    % read image and open net
    img = readimage(imds,n)
    net = alexnet; % req image of size 277x277x3
    
    % classify images
    [pred, scrs] = classify(net,img) % single prediction, all scores
    % [preds, scores] = classify(net,imds) % to do every image in folder, scores will be nx1000 matrix 
    ly = net.Layers; % extract layers from net
    inlayer = ly(1); % first layer
    insz = inlayer.InputSize; % size of first layer
    outlayer = ly(end) % last layer/output layer

    %% look at details of the classifications
    categorynames = outlayer.Classes; % classes property of the last layer
    
    % display bar graph of scores in classification
    thresh = median(scores) + std(scores); % based on being more than a std away
    highscores = scores > thresh;
    bar(scores(highscores))
    % xticks(1:length(scores(highscores)))
    xticklabels(categorynames(highscores))
    xtickangle(60)
    % max(scores,[],2) % to see all maxes' value
    
end
    
%% transfer learning

load pathToImages
load trainedFlowerNetwork flowernet info

%Create a network by modifying AlexNet
n = 24;
anet = alexnet;
layers = anet.Layers
fc = fullyConnectedLayer(numClasses) %create a new fully connected layer with n nodes
layers(end-2) = fc % replace last layer with new fc layer
layers(end) = classificationLayer

% set initial training weights with an initial conservative learning rate
% (weight changing step) 
    % ?stochastic gradient descent with momentum?.
opts = trainingOptions('sgdm', 'InitialLearnRate', 0.001)

%Perform training
[flowernet,info] = trainNetwork(trainImgs, layers, options);

%Use trained network to classify test images
testpreds = classify(flowernet,testImgs);

% show loss
load pathToImages
load trainedFlowerNetwork flowernet info
plot(info.TrainingLoss)
flwrPreds = classify(flowernet,testImgs)% testImgs is a datastore

% evaluate accuracy
load pathToImages.mat
pathToImages
flwrds = imageDatastore(pathToImages,'IncludeSubfolders',true,'LabelSource','foldernames');
[trainImgs,testImgs] = splitEachLabel(flwrds,0.98);
load trainedFlowerNetwork flwrPreds
flwrActual = testImgs.Labels
numCorrect = nnz(flwrPreds == flwrActual)
fracCorrect = numCorrect/24
confusionchart(testImgs.Labels, flwrPreds)


%% NOTES
% want loss and accuracy to both decrease as interations procede,
% okay if accuracy plateaus and loss keeps decr because means is still 
% picking same final choice but is more confident on prediction

 
 



