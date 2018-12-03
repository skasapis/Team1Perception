
% get path to every image
files = dir('deploy/test/*/*_image.jpg');

% alt method to get filepaths and may be better
imds = imageDatastore('deploy/test/*/*_image.jpg');
fnames = imds.Files; %names of all files

% loop through every image in test set -- crop to car and apply label
numImg = numel(files);
labels = zeros(numImg, 1);

for idx = 1:numImg
    
    % get folder and picture name
    snapshot = [files(idx).folder, '/', files(idx).name];
    fullName = snapshot(107:end); % wrt Izzy path
    %name = snapshot(??:end-10); % wrt Jen path
    
    % remove the "_image.jpg" for when printing to the file
    printName = fullName(1:end-10);
    
    croppedImg = cropImage(fnames(idx), snapshot);
    labels(idx) = assign_Label(croppedImg);
    printToFile(labels, printName);
    
end


%% /////////////////// SUPPLEMENTARY FUNCTIONS ///////////////////


function [croppedImg] = cropImage(imgPath, snapshot)
    % Perception part 1 code to find region in image containing the car
    img = readimage(imgPath,n)
    % determine bounding box
    
    % crop image - please scale to be 227x227 for use with AlexNet
    
end


function [label] = assign_Label(cropImg)
    % by taking Alexnet and training it on our data its called transfer
    % learning
    % apply NN to cropped image
    label = randi(3)-1;
    label = trainOnCroppedImg();
    
end

function [] = printToFile(labels, printName)
    % open file to print to
    fileID = fopen('Team1.txt','w'); % will have to change 'w' if want to append instead of overwrite
    fprintf(fileID,'guid/image,label\n');

    for n = 1:numel(labels)
        % print name of image
        fprintf(fileID,printName);

        % print comma label newline
        fprintf(fileID,',%d\n',labels(n));
    end
    
    % close file when done printing
    fclose(fileID);
end

% helpful functions
    % montage(imds) %place whole dataset into photo montage
