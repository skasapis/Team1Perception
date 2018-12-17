%Marie Brooks and Izzy Salley
%66% accuracy obtained 12/2/18
%REF: https://www.mathworks.com/help/deeplearning/examples/train-deep-learning-network-to-classify-new-images.html

%Confirmed that changing the fully connected and classification layers is an acceptable modification on 12/2

%% Create image datastore
all_imds=imageDatastore('deployCropped2/trainval','IncludeSubfolders',1,'FileExtensions','.jpg');
% Add labels from .csv
labels=dlmread('deploy/labels.csv',',',1,1);

%reformatting to make categorical possible
labels_str=cellstr(num2str(labels));
valueset={'0','1','2'};
labels_cat=categorical(labels_str,valueset);
all_imds.Labels=labels_cat;

%Split off some values for validation
[imdsTrain,imdsValidation] = splitEachLabel(all_imds,0.6,'randomized');

%% import googlenet and set options
net=googlenet;
% Choose layers to replace
lgraph = layerGraph(net);

%If using AlexNet:
%lgraph = layerGraph(net.Layers); 
%set FCC and classification layers as required

% Googlenet:
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

% layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

%Data augmentation to meet size requirement
inputSize = net.Layers(1).InputSize;
%Optional data augmentation code:
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
   'RandXReflection',true, ...
   'RandXTranslation',pixelRange, ...
   'RandYTranslation',pixelRange, ...
   'RandXScale',scaleRange, ...
   'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
   'DataAugmentation',imageAugmenter);

%comment out the following augimdsTrain line if above is
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

epochs = 5
% Set options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',100, ...
    'MaxEpochs',epochs, ... % would like to try 8 or 12
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',20);%, ...
    %'Verbose',false), ...
    %'Plots','training-progress');

%% Train network
tic
[new_net, info] = trainNetwork(augimdsTrain,lgraph,options);
toc
save new_net
save info
disp('TRAINING COMPLETE!');

%% classify test data
% Create augmented dataset from test data
test_imds=imageDatastore('deploy/test','IncludeSubfolders',1,'FileExtensions','.jpg');
augimdstest = augmentedImageDatastore(inputSize(1:2),test_imds);
%Classify test data
tic
[test_labels,~] = classify(new_net,augimdstest);
%Convert to 0,1,2
test_labels=grp2idx(test_labels)-1;
toc
disp('CLASSIFICATION COMPLETE!');

printToFile(test_labels);
disp('TEXT FILE GENERATED');

%% display Loss and Accuracy
figure(1)
subplot(2,1,1); plot(info.TrainingAccuracy,'b'); xlabel('Epoch'); ylabel('Accuracy');
hold on; plot(info.ValidationAccuracy, 'k-*'); grid on; axis([1 epochs 0 100]);

subplot(2,1,2); plot(info.TrainingLoss,'r'); xlabel('Epoch'); ylabel('Loss');
hold on; plot(info.ValidationLoss,'k-*'); grid on; axis([1 epochs 0 2]);
print('AccuracyAndLoss', '-dpng')


%% ////////////////// SUPPLEMENTARY FUNCTIONS //////////////////


function [printName] = getPrintName(idx)
    % get folder and picture name
    files = dir('deploy/test/*/*_image.jpg');
    snapshot = [files(idx).folder, '/', files(idx).name];
    %fullName = snapshot(107:end); % wrt Izzy path and Chris
    %fullName=snapshot(63:end); %wrt Marie path
    fullName=snapshot(47:end); %wrt ssh path
    
    % remove the "_image.jpg" for when printing to the file
    printName = fullName(1:end-10);
end

function [] = printToFile(labels)
    % open file to print to
    fileID = fopen('Team1_submission15.txt','w'); % will have to change 'w' if want to append instead of overwrite
    fprintf(fileID,'guid/image,label\n');
    
    for n = 1:numel(labels)
        % print name of image
        printName = getPrintName(n);
        fprintf(fileID,printName);
        % print comma label newline
        fprintf(fileID,',%d\n',labels(n));
    end
    
    % close file when done printing
    fclose(fileID);
end


