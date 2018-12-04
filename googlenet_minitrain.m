%Marie Brooks, mariegb
%Ran 12/1/18 with resulting 90% accuracy on single folder of UNCROPPED data. Took 6
%min 42 seconds to run on 1 CPU on 2011 macbook pro.
%REF: https://www.mathworks.com/help/deeplearning/examples/train-deep-learning-network-to-classify-new-images.html


% to run through ssh: matlab -nodisplay -nodesktop -r "run ./googlenet_minitrain.m"



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

%% import googlenet
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


epochs = 8;
%Set options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',100, ...
    'MaxEpochs',epochs, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Train network
[net, info] = trainNetwork(augimdsTrain,lgraph,options);
save net

figure(1)
subplot(2,1,1); plot(info.TrainingAccuracy,'b'); xlabel('Epoch'); ylabel('Accuracy');
hold on; plot(info.ValidationAccuracy, 'k-*'); grid on; axis([1 epochs 0 100]);

subplot(2,1,2); plot(info.TrainingLoss,'r'); xlabel('Epoch'); ylabel('Loss');
hold on; plot(info.ValidationLoss,'k-*'); grid on; axis([1 epochs 0 2]);
print('AccuracyAndLoss', '-dpng')


%Create augmented dataset from test data
test_imds=imageDatastore('deploy/test/0ff0a23e-5f50-4461-8ccf-2b71bead2e8e/*_image.jpg');
augimdstest = augmentedImageDatastore(inputSize(1:2),test_imds);
%% Classify test data
[test_labels,~] = classify(net,augimdstest);

printToFile(test_labels);
% fileID = fopen('stats.txt','w');

disp('CLASSIFICATION DONE!');



%% ////////////////// SUPPLEMENTARY FUNCTIONS //////////////////


function [printName] = getPrintName(idx)
    % get folder and picture name
    files = dir('deploy/test/*/*_image.jpg');
    snapshot = [files(idx).folder, '/', files(idx).name];
    %fullName = snapshot(107:end); % wrt Izzy path and Chris
    %fullName=snapshot(63:end); %wrt Marie path
    fullName=snapshot(40:end); %wrt ssh path
    
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


