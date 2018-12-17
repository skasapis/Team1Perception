
I = imread('deploy/test/0729ab2d-7fb0-4799-975f-c38fd350bf9c/0000_image.jpg');

detbbox = [375 196 299 182]
detbbox = [1    44   312   763]
detbbox = [700   215   877   471];
detbbox = [689   199   874   483];
detbbox = [25           1        1407         702];

I = insertShape(I, 'Rectangle', detbbox, 'Color', {'green'},'Opacity',0.7);
imshow(I)