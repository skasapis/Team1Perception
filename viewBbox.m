
I = imread('detectTest2.png');
figure(1)
imshow(I)

detbbox = [375 196 299 182]
detbbox = [361 380 297 162]

I = insertShape(I, 'Rectangle', detbbox, 'Color', {'green'},'Opacity',0.7);
figure(2)
imshow(I)