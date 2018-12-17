function [bboxAll, trainIdx] = BBox_Code(numTrain)

files = dir('deploy/trainval/*/*_image.jpg');

% index of the first image in a given folder
goodCarsFolder = [ones(1,12), 111, 422*ones(1,25)];

% image number + 1 because are labeled starting at "0000.png"
goodCarsFile = [54 74 80 88 89 92 95 100 103 107 109 110, ... % folder 1
                1, ... %folder 2
                6 7 8 12 48 49 85 87 88 89 103 104 105 107 109 111 122 123 184 194 199 200 213 220 232,... % folder 3
                ];
goodCarsFile = goodCarsFile + goodCarsFolder - 1;
numGood = numel(goodCarsFile);

count = 1;
skip = false;
for idx = 1:numTrain %numel(files)

    if numTrain <= numGood
        snapshot = [files(goodCarsFolder(idx)).folder, '/', files(goodCarsFile(idx)).name];
        disp(snapshot)
    else
        snapshot = [files(idx).folder, '/', files(idx).name];
        disp(snapshot)
    end

%     img = imread(snapshot);

    xyz = read_bin(strrep(snapshot, '_image.jpg', '_cloud.bin'));
    xyz = reshape(xyz, [], 3)';

    proj = read_bin(strrep(snapshot, '_image.jpg', '_proj.bin'));
    proj = reshape(proj, [4, 3])';

    try
        bbox = read_bin(strrep(snapshot, '_image.jpg', '_bbox.bin'));
    catch
        disp('[*] no bbox found.')
        bbox = single([]);
        bboxNew = single([]);
        skip = true
    end
    bbox = reshape(bbox, 11, [])';

    uv = proj * [xyz; ones(1, size(xyz, 2))];
    uv = uv ./ uv(3, :);

    for k = 1:size(bbox, 1)
        R = rot(bbox(k, 1:3));
        t = reshape(bbox(k, 4:6), [3, 1]);

        sz = bbox(k, 7:9);
        [vert_3D, edges] = get_bbox(-sz / 2, sz / 2);
        vert_3D = R * vert_3D + t;
    %proj is 3x4 instrinsic camera matrix
        vert_2D = proj * [vert_3D; ones(1, size(vert_3D, 2))];
        %homogenizing 
        vert_2D = vert_2D ./ vert_2D(3, :);
        [X,I] = min(vert_2D(1,:));
        [Y,I] = max(vert_2D(2,:));
        [TRx,I] = max(vert_2D(1,:));
        [TRy,I] = max(vert_2D(2,:));
        [BRy,I] = min(vert_2D(2,:));
        [BRx,I] = max(vert_2D(1,:));
        [BLy,I] = min(vert_2D(2,:));
        [BLx,I] = min(vert_2D(1,:));
        x_length = TRx-X;
        y_length = Y-BLy;

        Y = Y-y_length;
        % upper left corner (X,Y)
        bboxNew = [X,Y,x_length,y_length];
    end
    
    if skip == true
        % dont include in train set because has no car
    elseif numTrain < numGood
        bboxAll{count} = bboxNew;
        trainIdx(count) = goodCarsFile(idx);
        count = count + 1;
    else
        bboxAll{count} = bboxNew;
        trainIdx(count) = idx;
        count = count + 1;
    end
end
% return
end

%% TO VISUALIZE WHERE OUR BOUNDING BOX IS -- VALIDATION
% imshow(img);
% hold on;
% % lower left corner, width height
% rectangle('Position',[X Y-y_length x_length y_length],...
%           'EdgeColor', 'c',...
%           'Curvature',[0.8,0.4],...
%           'LineWidth',2,...
%           'LineStyle','-')
      


%% /////////////// SUPPLEMENTARY FUNCTIONS ///////////////

function [v, e] = get_bbox(p1, p2)
v = [p1(1), p1(1), p1(1), p1(1), p2(1), p2(1), p2(1), p2(1)
    p1(2), p1(2), p2(2), p2(2), p1(2), p1(2), p2(2), p2(2)
    p1(3), p2(3), p1(3), p2(3), p1(3), p2(3), p1(3), p2(3)];
e = [3, 4, 1, 1, 4, 4, 1, 2, 3, 4, 5, 5, 8, 8
    8, 7, 2, 3, 2, 3, 5, 6, 7, 8, 6, 7, 6, 7];
end

%%
function R = rot(n)
theta = norm(n, 2);
if theta
  n = n / theta;
  K = [0, -n(3), n(2); n(3), 0, -n(1); -n(2), n(1), 0];
  R = eye(3) + sin(theta) * K + (1 - cos(theta)) * K^2;
else
  R = eye(3);
end
end

%%
function data = read_bin(file_name)
id = fopen(file_name, 'r');
data = fread(id, inf, 'single');
fclose(id);
end
