IMDS = imageDatastore('D:\Cultural Fest','IncludeSubfolders',true,'LabelSource','foldernames');
[trainingSet,testSet] = splitEachLabel(IMDS,70);

%%

trainingSetLength = length(trainingSet.Files);
training_features = [];
for loop=1:trainingSetLength
    im = imread(trainingSet.Files{loop});
    feature = HOG(im);
    training_features = [training_features,feature];
end

%%

testSetLength = length(testSet.Files);
test_features = [];
for loop=1:testSetLength
    im = imread(testSet.Files{loop});
    feature = HOG(im);
    test_features = [test_features,feature];
end

%%
trainingLabel = trainingSet.Labels;
testLabel = testSet.Labels;
SupportVector = fitcecoc(training_features,trainingLabel,'Learners','Linear','Coding','onevsall','ObservationsIn','colusmns');
output = predict(SupportVector,test_features');
%%

variable = testLabel==output;
count = 0;
testLabelLength = length(testLabel);

for loop = 1:testLabelLength
    if( variable(loop) == 1)
        count=count+1;
    end
end

percentage = (count/testLabelLength)*100;

%%
disp ('The accuracy of data set using HOG features is');
disp(percentage);
