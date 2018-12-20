test_labelsCrop = [1 0 1 0 1 0];
test_labelsFull = [1 0 0 1 2 2];

cropZeros = sum(test_labelsCrop == 0)
fullZeros = sum(test_labelsFull == 0)

cropZeroID = find(test_labelsCrop == 0);
fullZeroID = find(test_labelsFull == 0);

disagreeOnZero = find(test_labelsFull(cropZeroID) ~= test_labelsCrop(cropZeroID));
relabelID = cropZeroID(disagreeOnZero);
test_labels(relabelID) = test_labelsFull(relabelID)



%%
y = [1 2 5; 1 2 1; 3 1 1; 1 2 1;]/10;

[m1, i1] = max(y')

z = [1 5 1; 5 1 1; 1 1 1; 1 5 1;]/10;

[m2, i2] = max(z')

cropUncertain = find(m1 < 0.6);
fullHigher = find(m2(cropUncertain) > m1(cropUncertain))
numReplaced = numel(fullHigher)
relabelID = cropUncertain(fullHigher)
test_labels(relabelID) = i2(relabelID) - 1;

x = 1;

