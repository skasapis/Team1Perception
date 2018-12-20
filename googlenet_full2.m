%% Load network
tic
load net4Crop
load net4Full
toc
disp('NETWORK LOADED!');

%% classify test data
% set dependencies
printTxtNm = 'Team1_submission.txt';
cropFldrNm = 'CroppedPics2';

%% load image dataset
test_imdsCrop=imageDatastore(cropFldrNm,'IncludeSubfolders',1,'FileExtensions','.jpg');
test_imdsFull=imageDatastore('deploy/test','IncludeSubfolders',1,'FileExtensions','.jpg');
% Create augmented dataset from test data - scale images to 224x224 for googlenet requirement
augimdstestCrop = augmentedImageDatastore(inputSize(1:2),test_imdsCrop);
augimdstestFull = augmentedImageDatastore(inputSize(1:2),test_imdsFull);

% decide which images should use crop vs full image for classification
tic
cropIdx = [];
fullIdx = [];
for idx = 1:numel(test_imdsCrop.Files)
    I = imread(test_imdsCrop.Files{idx});
    [w h d] = size(I);
    if w < 20 || h < 20
        fullIdx = [fullIdx, idx];
    else
        cropIdx = [cropIdx, idx];
    end
end

%Classify test data
[test_labelsCrop, scoresCrop] = classify(net4Crop,augimdstestCrop);
[test_labelsFull, scoresFull] = classify(net4Full,augimdstestFull);

%Convert to 0,1,2
% the classify function returns labels of 1,2,3 so we must subtract 1
test_labelsCrop=grp2idx(test_labelsCrop)-1;
test_labelsFull=grp2idx(test_labelsFull)-1;

% fill in cooresponding index for cropped and full within the whole set
test_labels(cropIdx) = test_labelsCrop(cropIdx);
test_labels(fullIdx) = test_labelsFull(fullIdx);


% edit index where full has higher score than cropped
% take max of each column of cropped, if max score less than 0.6 then check
% if the max score in full is higher than cropped score

[m1, i1] = max(scoresCrop');
[m2, i2] = max(scoresFull');

cropUncertain = find(m1 < 0.6);
fullHigher = find(m2(cropUncertain) > m1(cropUncertain));
numReplaced = numel(fullHigher);
relabelID = cropUncertain(fullHigher);
test_labels(relabelID) = i2(relabelID) - 1;


% edit index where cropped claims label of 0 but full doesn't
cropZeros = sum(test_labelsCrop == 0);
fullZeros = sum(test_labelsFull == 0);

cropZeroID = find(test_labelsCrop == 0);
fullZeroID = find(test_labelsFull == 0);

disagreeOnZero = find(test_labelsFull(cropZeroID) ~= test_labelsCrop(cropZeroID));
relabelID = cropZeroID(disagreeOnZero);
test_labels(relabelID) = test_labelsFull(relabelID);

toc
disp('CLASSIFICATION COMPLETE!');

printToFile(test_labels, cropFldrNm, printTxtNm);
disp('TEXT FILE GENERATED');


%% ////////////////// SUPPLEMENTARY FUNCTIONS //////////////////


function [printName] = getPrintName(idx,cropFldrNm)
    % get folder and picture name
    files = dir([cropFldrNm, '/*/*_image.jpg']);
    snapshot = [files(idx).folder, '/', files(idx).name];
    % later learned all image folders have same length
    fullName = snapshot(end-50:end);
    % remove "_image.jpg" for when printing to the file
    printName = fullName(1:end-10);
end

function [] = printToFile(labels, cropFldrNm, printTxtNm)
    % open file to print to
    fileID = fopen(printTxtNm,'w'); % will have to change 'w' if want to append instead of overwrite
    
    fprintf(fileID,'guid/image,label\n');%line 1
    for n = 1:numel(labels)
        % print name of image
        printName = getPrintName(n,cropFldrNm);
        fprintf(fileID,printName);
        % print comma label newline
        fprintf(fileID,',%d\n',labels(n));
    end
    
    % close file when done printing
    fclose(fileID);
end


