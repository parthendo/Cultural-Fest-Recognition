close all
imds=imageDatastore('D:\Cultural Fest\Event','includeSubfolders',true,'labelSource','foldernames');
[training_set,testing_set]=splitEachLabel(imds,0.6);

disp('part1');
%%
len=length(training_set.Files);
training_features=[];
for k=1:len
    im1=imread(training_set.Files{k});
    im=rgb2gray(im1);
    h2=extractLBPFeatures(im);
    training_features=[training_features;h2];
end
disp('part2');
%%
len=length(testing_set.Files);
testing_features=[];
for l=1:len
    img1=imread(testing_set.Files{l});
    img=rgb2gray(img1);
    h2=extractLBPFeatures(img);
    testing_features=[testing_features;h2];
end
disp('part 3');
%%

training_label=training_set.Labels;
test_label=testing_set.Labels;
sv=fitensemble(training_features,training_label', 'Bag',100,'Tree','Type', 'classification');
out=predict(sv,testing_features);

disp('part 4');
%%
accu=accuracy(test_label,out);
disp('accuracy=');
disp(accu);