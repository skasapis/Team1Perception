%Marie Brooks, mariegb
%Ran 12/1/18 with resulting 90% accuracy on single folder of UNCROPPED data. Took 6
%min 42 seconds to run on 1 CPU on 2011 macbook pro.
%REF: https://www.mathworks.com/help/deeplearning/examples/train-deep-learning-network-to-classify-new-images.html

%I believe we need to change something in this network, such as
%adding/deleting layers or changing an activation function. I don't think
%just changing the fully connected and classification layers is enough.

% Create image datastore from one folder of training images, for local run
% purposes.
all_imds=imageDatastore('deploy/trainval','IncludeSubfolders',1,'FileExtensions','.jpg');
% Add labels from .csv
%labels=dlmread('deploy/labels.csv',',',[1 1 110 1]);
labels=dlmread('deploy/labels.csv',',',1,1);

labels_str=cellstr(num2str(labels)); %reformatting to make categorical possible
valueset={'0','1','2'};
labels_cat=categorical(labels_str,valueset);
all_imds.Labels=labels_cat;
%Split off some values for validation
[imdsTrain,imdsValidation] = splitEachLabel(all_imds,0.7);

%import googlenet
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

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

%Data augmentation to meet size requirement
inputSize = net.Layers(1).InputSize;
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

%Set options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',100, ...
    'MaxEpochs',3, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

%Train network
new_net = trainNetwork(augimdsTrain,lgraph,options);

%Create augmented dataset from test data
test_imds=imageDatastore('deploy/test','IncludeSubfolders',1,'FileExtensions','.jpg');
augimdstest = augmentedImageDatastore(inputSize(1:2),test_imds);
%Classify test data
[test_labels,~] = classify(new_net,augimdstest);

printToFile(test_labels);

function [printName] = getPrintName(idx)
    % get folder and picture name
    files = dir('deploy/test/*/*_image.jpg');
    snapshot = [files(idx).folder, '/', files(idx).name];
    %fullName = snapshot(107:end); % wrt Izzy path
    %name = snapshot(??:end-10); % wrt Jen path
    fullName=snapshot(63:end); %wrt Marie path
    % remove the "_image.jpg" for when printing to the file
    printName = fullName(1:end-10);
end

function [] = printToFile(labels)
    % open file to print to
    fileID = fopen('Team1.txt','w'); % will have to change 'w' if want to append instead of overwrite
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


