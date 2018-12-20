
%% EDIT THIS LINE TO COORESPOND WITH THE DETECTOR THAT PRODUCED CroppedPics2 or 3 -- whichever is better
load cdetector % *** edit
disp('DETECTOR LOADED');
    
%% APPLY DETECTOR TO TEST IMAGES
testds = imageDatastore('deploy/test','IncludeSubfolders',1,'FileExtensions','.jpg');

printTxtNm = 'Team1_centroids.txt';
fileID = fopen(printTxtNm,'w');
fprintf(fileID,'guid/image/axis,value\n');%line 1


tic
for idx = 1:numel(testds.Files)

    % Read the image
    snapshot = testds.Files{idx};
    
    % get size of original image
    I = imread(snapshot);    
    [h w d] = size(I);
    xcenter = round(w/2);
    ycenter = round(h/2);
    
    % Run the detector
    [detbbox, scores, labels] = detect(cdetector,I);
    
    % crop image and save
    if numel(detbbox < 4) == 0
        % save original image -- no cropping
        box = [1 1 w-1 h-1];
    else       
        [mx, maxScoreIdx] = max(scores);
        box = detbbox(maxScoreIdx,1:4);

    end
    
    % calculate center of bbox
    xbox = box(3)/2+box(1);
    ybox = h - (box(4)/2+box(2)); % need larger values in top half of img
    
    % get point cloud
    xyz = read_bin(strrep(snapshot, '_image.jpg', '_cloud.bin'));
    xyz = reshape(xyz, [], 3)';
    
    % calculate centroid of car
    mxs = max(xyz');
    mns = min(xyz');
    
    xmx = mxs(1);
    xmn = mns(1);
    ymx = mxs(2);
    ymn = mns(2);
    zmx = mxs(3);
    zmn = mns(3);
    
    x = ((xbox - xcenter)/w)*(xmx-xmn);
    y = ((ybox - ycenter)/h)*(ymx-ymn);
    z = (zmx-zmn)/2;
    
    
    % SAVE RESULTS
    name = testds.Files(idx);
    name = name{1}(end-50:end-10);
    
    % print x
    fprintf(fileID,[name, '/x']);    
    fprintf(fileID,',%d\n', x);
    % print y
    fprintf(fileID,[name, '/y']);    
    fprintf(fileID,',%d\n', y);
    
    % print z
    fprintf(fileID,[name, '/z']);    
    fprintf(fileID,',%d\n', z);

end

disp('CENTROIDS FOUND AND PRINTED')
toc

fclose(fileID);







%%
function data = read_bin(file_name)
id = fopen(file_name, 'r');
data = fread(id, inf, 'single');
fclose(id);
end
