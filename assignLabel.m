
% get path to every image
files = dir('deploy/test/*/*_image.jpg');


% loop through every image in test set -- crop to car and apply label
numImg = numel(files);
labels = zeros(numImg, 1);

for idx = 1:numImg
    
    % get folder and picture name
    snapshot = [files(idx).folder, '/', files(idx).name];
    fullName = snapshot(87:end); % wrt Izzy path
    %name = snapshot(??:end-10); % wrt Jen path
    
    % remove the "_image.jpg" for when printing to the file
    print_name = fullName(1:end-10);
    
    croppedImg = cropImage(Img, snapshot);
    labels(idx) = assign_Label(croppedImg);
    printToFile(labels, printName);
    
end


%% /////////////////// SUPPLEMENTARY FUNCTIONS ///////////////////


function [croppedImg] = cropImage(imgPath)
    % Perception part 1 code to find region in image containing the car
    img = imread(imgPath);
    % determine bounding box
    bBox = [1 1 10 10; 1 10 1 10]; % x of corners; y of corners
    
    % crop image
    croppedImg = img(bBox(1,1):bBox(3,1), bBox(2,1):bBox(2,3));
end


function [label] = assign_Label(cropImg)
    % apply NN to cropped image
    label = randi(3)-1;
    
end

function [] = printToFile(labels, folder_plus_Img)
    % open file to print to
    fileID = fopen('Team1.txt','w'); % will have to change 'w' if want to append
    fprintf(fileID,'guid/image,label\n');

    for n = 1:numel(labels)
        % print name of image
        fprintf(fileID,folder_plus_Img);

        % print comma label newline
        fprintf(fileID,',%d\n',labels(n));
    end
    
    % close file when done printing
    fclose(fileID);
end
