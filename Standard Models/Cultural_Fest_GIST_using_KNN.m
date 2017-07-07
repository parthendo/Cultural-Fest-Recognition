close all
imds=imageDatastore('D:\Cultural Fest\Event','includeSubfolders',true,'labelSource','foldernames');
[training_set,testing_set]=splitEachLabel(imds,0.6);


%%
g=length(training_set.Files);
training_features=[];
for k=1:g
    im1=imread(training_set.Files{k});
    clear param
param.imageSize = [256 256]; % it works also with non-square images
param.orientationsPerScale = [8 8 8 8];
param.numberBlocks = 4;
param.fc_prefilt = 4;

% Computing gist requires 1) prefilter image, 2) filter image and collect
% output energies
[gist1, param] = LMgist(im1, '', param);

    training_features=[training_features;gist1];
end
%%
g=length(testing_set.Files);
testing_features=[];
for l=1:g
    im1=imread(testing_set.Files{l});
    % Parameters:
clear param
param.imageSize = [256 256]; % it works also with non-square images
param.orientationsPerScale = [8 8 8 8];
param.numberBlocks = 4;
param.fc_prefilt = 4;

% Computing gist requires 1) prefilter image, 2) filter image and collect
% output energies
[gist1, param] = LMgist(im1, '', param);

    testing_features=[testing_features;gist1];
end
%%
training_label=training_set.Labels;
test_label=testing_set.Labels;
sv=fitcknn(training_features,training_label,'NumNeighbors',5);
out=predict(sv,testing_features);
%%
accu=accuracy(test_label,out);
disp('accuracy=');
disp(accu);