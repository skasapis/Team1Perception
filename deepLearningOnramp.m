

imds = imageDatastore('deploy/test/*/*_image.jpg');
fname = imds.Files %names of all files
f1 = fname{1}

%%

for n = 1:numel(fname)
    % read image and open net
    img = readimage(imds,n)
    net = alexnet; % req image of size 277x277x3
    
    % classify images
    [pred, scrs] = classify(net,img) % single prediction, all scores
    % [preds, scores] = classify(net,imds) % to do every image in folder, scores will be nx1000 matrix 
    ly = net.Layers; % extract layers from net
    inlayer = ly(1); % first layer
    insz = inlayer.InputSize; % size of first layer
    outlayer = ly(end) % last layer/output layer

    %% look at details of 
    categorynames = outlayer.Classes; % classes property of the last layer
    
    % display bar graph of scores in classification
    thresh = median(scores) + std(scores); % based on being more than a std away
    highscores = scores > thresh;
    bar(scores(highscores))
    % xticks(1:length(scores(highscores)))
    xticklabels(categorynames(highscores))
    xtickangle(60)
    % max(scores,[],2) % to see all maxes' value
    
end
    