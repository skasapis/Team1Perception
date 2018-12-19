%% Load network
tic
load net4Crop
load net4fullSz
toc
disp('NETWORK LOADED!');

%% classify test data
% set dependencies
printTxtNm = 'Team1_submissionX.txt';
cropFldrNm = 'CroppedPics2';

% load image dataset
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
    if w < 30 || h < 30 || w > 1000 || h > 800
        fullIdx = [fullIdx, idx];
    else
        cropIdx = [cropIdx, idx];
    end
end

%Classify test data
[test_labelsCrop, scoresCrop] = classify(net4Crop,augimdstestCrop);
[test_labelsFull, scoresFull] = classify(net4Full,augimdstestFull);

% scores gives you the score for each label 0,1,2 -- if you want to not classify
% anything of score 0 you could then assign the label of the next highest
% score ( 0 consists of 7% of our training data -- seems possible it is
% also a very small percentage of our test data)

%Convert to 0,1,2
% the classify function returns labels of 1,2,3 so we must subtract 1
test_labelsCrop=grp2idx(test_labelsCrop)-1;
test_labelsFull=grp2idx(test_labelsFull)-1;

% fill in cooresponding index for cropped and full within the whole set
test_labels(cropIdx) = test_labelsCrop;
test_labels(fullIdx) = test_labelsFull;

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
    printName = snapshot(end-50:end);
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


