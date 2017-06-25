IMDS = imageDatastore('D:\Cultural Fest','IncludeSubfolders',true,'LabelSource','foldernames');
[trainingSet,testSet] = splitEachLabel(IMDS,70);
trainingSetLength = length(trainingSet.Files);
testSetLength = length(testSet.Files);
training_features = [];
test_features = [];


