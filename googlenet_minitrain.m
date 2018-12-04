%Marie Brooks, mariegb
%Ran 12/1/18 with resulting 90% accuracy on single folder of UNCROPPED data. Took 6
%min 42 seconds to run on 1 CPU on 2011 macbook pro.
%REF: https://www.mathworks.com/help/deeplearning/examples/train-deep-learning-network-to-classify-new-images.html

% Create image datastore from one folder of training images, for local run
% purposes.
mini_imds=imageDatastore('deploy/trainval/047b864f-0753-448b-9483-f990ae41abaf/*_image.jpg');

% Add labels from .csv
labels=dlmread('deploy/labels.csv',',',[1 1 110 1]);
labels_str=cellstr(num2str(labels)); %reformatting to make categorical possible
valueset={'0','1','2'};
labels_cat=categorical(labels_str,valueset);
mini_imds.Labels=labels_cat;

%Split off some values for validation
[imdsTrain,imdsValidation] = splitEachLabel(mini_imds,0.7);

%import googlenet
net=googlenet;
% Choose layers to replace
lgraph = layerGraph(net);
learnableLayer=lgraph.Layers(142);
classLayer=lgraph.Layers(144);

%Replace fully connected layer
numClasses=3;
newLearnableLayer=fullyConnectedLayer(numClasses,'Name','new_fc',...
    'WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

%Replace class layer
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

%Freeze initial layers - optional, may speed things up
layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

%Data augmentation to meet size requirement
inputSize = net.Layers(1).InputSize;
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

%Set options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',100, ...
    'MaxEpochs',8, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

%Train network
[net, info] = trainNetwork(augimdsTrain,lgraph,options);

figure(1)
subplot(2,1,1); plot(info.TrainingAccuracy,'b'); ylabel('Accuracy');
hold on
plot(info.ValidationAccuracy, 'k--*');
subplot(2,1,2); plot(info.TrainingLoss,'r');ylabel('Loss');
plot(info.ValidationLoss,'k--*');
print('AccuracyAndLoss', '-dpng')


%Still need to write the part where it classifies the test data and writes
%it in our required format.

disp('CLASSIFICATION DONE!');

fileID = fopen('Team1.txt','w');
fprintf(fileID,'trained network by classification has not been set up\n');

