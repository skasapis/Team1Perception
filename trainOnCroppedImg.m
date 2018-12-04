%% Izzy Salley
% 12/1/2018

%% transfer learning
% http://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html
% https://matlabacademy.mathworks.com

splitFldrBool = 1;
[trainds, testds, numTrain, numTest] = fillDatastores(splitFldrBool);

retrainBool = 0;
carNet = getNet(trainds, retrainBool);

%classify test image(s)
singleBool = 0;
if singleBool
    idx = 1;
    img = readimage(testds,idx);
    [carPreds, scores] = classify(carNet, img);
    
    % display bar graph of scores in classification
    figure(2)
    categorynames = [{'0'},{'1'},{'2'}];
    b = bar(scores,'FaceColor','flat');
    [m, maxIdx] = max(scores);
    b.CData(maxIdx,:) = [0 0.8 0.8];
    xticklabels(categorynames)
    grid on
    
else
    [carPreds, scores] = classify(carNet, testds);
    printToFile(carPreds)
    
    figure(2)
    confusionchart(testds.Labels, carPreds)
    
    % print statistics
    carActual = testds.Labels;
    numCorrect = nnz(carPreds == carActual)
    fracCorrect = numCorrect/24
   
end





%% ////////////////// SUPPLEMENTARY FUNCTIONS //////////////////

function [trainds, testds, numTrain, numTest] = fillDatastores(splitFldrBool)
    % add images to datastore
        % I had to pick one that started with a letter to the because
        % matlab doesn't like names that start with a number
    trainDS = imageDatastore('deploy/trainval/croppedTrainAlex/*_image.jpg');
    fname = trainDS.Files; %names of all files, to access f1 = fname{1}
    numTrain = numel(fname);

    % Add labels to training datastore
    groundTruth = csvread('deploy/trainval/labels.csv',1,1);
    groundTruth = groundTruth(1:numTrain);
    labels_str=cellstr(num2str(groundTruth));
    valueset={'0','1','2'};
    labels_cat=categorical(labels_str,valueset);
    trainDS.Labels = labels_cat;

    if splitFldrBool
    % split single training folder for now
        [trainds,testds] = splitEachLabel(trainDS,0.7,'randomized');
        fname = trainDS.Files; %names of all files, to access f1 = fname{1}
        tname = testds.Files;
        numTrain = numel(fname);t
        numTest = numel(tname);
    else
        testds = imageDatastore('deploy/test/croppedTestAlex/*_image.jpg'); %d7f7c063-df18-4f96-a071-aa634b9e502e
        trainds = trainDS;
        fname = trainds.Files; %names of all files, to access f1 = fname{1}
        tname = testds.Files;
        numTrain = numel(fname);
        numTest = numel(tname);
    end
end



function [net] = getNet(trainds, retrainBool)
% either load pretrained custom net or retrain a net as desired

if retrainBool
    %Create a network by modifying AlexNet via transfer learning
    numClasses = 3;
    net = alexnet;
    % net=googlenet;
    layers = net.Layers;
    fc = fullyConnectedLayer(numClasses); %create a new fully connected layer with n nodes
    layers(end-2) = fc; % replace layer with new fc layer
    layers(end) = classificationLayer;

    % set initial training weights with an initial conservative learning rate
    % (weight changing step) 
        % ?stochastic gradient descent with momentum?.
    options = trainingOptions('sgdm', 'InitialLearnRate', 0.001);

    % options = trainingOptions('sgdm', ...
    %     'MiniBatchSize',100, ...
    %     'MaxEpochs',6, ...
    %     'InitialLearnRate',3e-4, ...
    %     'Shuffle','every-epoch', ...
    %     'ValidationData',augimdsValidation, ...
    %     'ValidationFrequency',3, ...
    %     'Verbose',false, ...
    %     'Plots','training-progress');

    %Perform training
    tic
    [net, info] = trainNetwork(trainds, layers, options);
    toc
    
    figure(1)
    subplot(2,1,1); plot(info.TrainingAccuracy,'b'); xlabel('Epoch'); ylabel('Accuracy');
    hold on; plot(info.ValidationAccuracy, 'k--*'); grid on; axis([0 epochs 0 100]);

    subplot(2,1,2); plot(info.TrainingLoss,'r'); xlabel('Epoch'); ylabel('Loss');
    hold on; plot(info.ValidationLoss,'k--*'); grid on; axis([0 epochs 0 4]);
    print('AccuracyAndLoss', '-dpng')
    
else % use preexisting custom net
    load carNet
    net = carNet;
    
end

end




function [] = printToFile(labels)
    % open file to print to
    fileID = fopen('Team1.txt','w'); % will have to change 'w' if want to append instead of overwrite
    fprintf(fileID,'guid/image,label\n');

    for n = 1:numel(labels)
        % print name of image
        printName = getPrintName(n)
        fprintf(fileID,printName);

        % print comma label newline
        fprintf(fileID,',%d\n',labels(n));
    end
    
    % close file when done printing
    fclose(fileID);
end
function [printName] = getPrintName(idx)
    % get folder and picture name
    files = dir('deploy/trainval/fc26f4db-22c1-49c7-be8d-b769476cdff2/*_image.jpg');
    snapshot = [files(idx).folder, '/', files(idx).name];
    fullName = snapshot(107:end); % wrt Izzy path
    %name = snapshot(??:end-10); % wrt Jen path
    
    % remove the "_image.jpg" for when printing to the file
    printName = fullName(1:end-10);
    
end
