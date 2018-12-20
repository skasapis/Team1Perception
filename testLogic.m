test_labelsCrop = [1 0 1 0 1 0];
test_labelsFull = [1 0 0 1 2 2];

cropZeros = sum(test_labelsCrop == 0)
fullZeros = sum(test_labelsFull == 0)

cropZeroID = find(test_labelsCrop == 0);
fullZeroID = find(test_labelsFull == 0);

disagreeOnZero = find(test_labelsFull(cropZeroID) ~= test_labelsCrop(cropZeroID));
relabelID = cropZeroID(disagreeOnZero);
test_labels(relabelID) = test_labelsFull(relabelID)

x = 1;