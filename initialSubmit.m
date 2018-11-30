files = dir('deploy/test/*/*_image.jpg');

fileID = fopen('Team1.txt','w');
fprintf(fileID,'guid/image,label\n')

for idx = 1:numel(files)
    snapshot = [files(idx).folder, '/', files(idx).name];
    % get folder and picture name
    name = snapshot(87:end-10);
    % print name of image
    fprintf(fileID,name);
    % pick rand number 0-2
    label = randi(1,1);
    % print comma rand number newline
    fprintf(fileID,',%d\n',label);
end

fclose(fileID);