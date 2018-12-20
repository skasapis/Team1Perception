function [printName] = getPrintName(idx)
    % get folder and picture name
    files = dir('deploy/test/*/*_image.jpg');
    snapshot = [files(idx).folder, '/', files(idx).name];
    %fullName = snapshot(107:end); % wrt Izzy path and Chris
    %fullName=snapshot(63:end); %wrt Marie path
    fullName=snapshot(47:end); %wrt ssh path
    
    % remove the "_image.jpg" for when printing to the file
    printName = fullName(1:end-10);
end