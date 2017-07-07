close all
imds=imageDatastore('D:\Cultural Fest\Event','includeSubfolders',true,'labelSource','foldernames');
[training_set,testing_set]=splitEachLabel(imds,0.6);

%%
len=length(training_set.Files);
training_features=[];
for loop=1:len
    im=imread(training_set.Files{loop});
    training_features=[training_features,HOG(im)];
end
%%
len=length(testing_set.Files);
testing_features=[];
for loop=1:len
    im=imread(testing_set.Files{loop});
    testing_features=[testing_features,HOG(im)];
end
%%

training_label=training_set.Labels;
test_label=testing_set.Labels;
sv=fitensemble(training_features',training_label, 'Bag',100,'Tree','Type', 'classification');
out=predict(sv,testing_features');
%%
accu=accuracy(test_label,out);
disp('accuracy=');
disp(accu);