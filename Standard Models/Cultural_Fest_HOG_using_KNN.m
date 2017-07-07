close all
imds=imageDatastore('D:\Cultural Fest\Event','includeSubfolders',true,'labelSource','foldernames');
[training_set,testing_set]=splitEachLabel(imds,0.6);
%%
len=length(training_set.Files);
training_features=[];
for k=1:len
    im=imread(training_set.Files{k});
    training_features=[training_features,HOG(im)];
end
%%
len=length(testing_set.Files);
testing_features=[];
for l=1:len
    im=imread(testing_set.Files{l});
    testing_features=[testing_features,HOG(im)];
end
%%

training_label=training_set.Labels;
test_label=testing_set.Labels;
sv=fitcknn(training_features',training_label,'NumNeighbors',5);
out=predict(sv,testing_features');
%%
accu=accuracy(test_label,out);
disp('accuracy=');
disp(accu);