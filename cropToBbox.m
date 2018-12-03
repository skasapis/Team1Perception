% load all images in a folder
% scale to 227x227
% save cropped images to croppedTrainAlex folder

trainds = imageDatastore('deploy/trainval/fc26f4db-22c1-49c7-be8d-b769476cdff2/*_image.jpg'); %223 pictures

fname = trainds.Files;
numTrain = numel(fname);

% files = dir('deploy/trainval/fc26f4db-22c1-49c7-be8d-b769476cdff2/*_image.jpg');


for n = 1:numTrain   
    snapshot = [files(n).folder, '/', files(n).name];

%     corners = get_bbox(snapshot);
    
    orig = readimage(trainds,n);
    img = imresize(orig,[227 227]);
%     img = orig(corners(1),corners(2),corners(3),corners(4));
    figure(1); subplot(2,1,1); imshow(orig)
    subplot(2,1,2); imshow(img)
    
    % save image
    filename = strrep(snapshot, 'deploy/trainval/fc26f4db-22c1-49c7-be8d-b769476cdff2/', 'deploy/trainval/croppedTrainAlex/');
    imwrite(img,filename)
    
    
end


%% ////////////////// SUPPLEMENTARY FUNCTIONS //////////////////

% not working yet
function [corners] = get_bbox(snapshot)

% get name for bbox file
try
    bbox = read_bin(strrep(snapshot, '_image.jpg', '_bbox.bin'));
catch
    disp('[*] no bbox found.')
    bbox = single([]);
end
bbox = reshape(bbox, 11, [])';

% convert bounding box into corners for cropping
for k = 1:size(bbox, 1)
    R = rot(bbox(k, 1:3));
    t = reshape(bbox(k, 4:6), [3, 1]);

    sz = bbox(k, 7:9);
    [vert_3D, edges] = get_bbox(-sz / 2, sz / 2);
    vert_3D = R * vert_3D + t;

    vert_2D = proj * [vert_3D; ones(1, size(vert_3D, 2))];
    vert_2D = vert_2D ./ vert_2D(3, :);

    clr = colors(mod(k - 1, size(colors, 1)) + 1, :);
    for i = 1:size(edges, 2)
        e = edges(:, i);

        figure(1)
        plot(vert_2D(1, e), vert_2D(2, e), 'color', clr)
    end
end

end

% FOLDERS WITH VARIABLILITY IN CLASS
% 215
% fc26
% f075