function [] = printToFile(labels)
    % open file to print to
    fileID = fopen('Team1_submission23.txt','w'); % will have to change 'w' if want to append instead of overwrite
    fprintf(fileID,'guid/image,label\n');
    
    for n = 1:numel(labels)
        % print name of image
        printName = getPrintName(n);
        fprintf(fileID,printName);
        % print comma label newline
        fprintf(fileID,',%d\n',labels(n));
    end
    
    % close file when done printing
    fclose(fileID);
end
