clear,clc
image_info = dir ('./../../data/');
image_dir = image_info.folder; 
image_names = {image_info.name};
image_names = image_names(~ismember(image_names, {'.', '..'})); 
image_numel = length(image_names);
opts.alpha = 1e-2;
opts.batchsize = 32;
opts.numepochs = 200;
opts.lambda = 0.0001; %weight decay
opts.momentum = .95;
opts.mom = 0.5;
opts.momIncrease = 20;

for i = 1:image_numel
    Train = load(fullfile(image_dir,image_names{1, i},'images.mat'));
    Test = load(fullfile(image_dir,image_names{1, i},'test.mat'));
    Info = load(fullfile(image_dir,image_names{1, i},'info.mat'));
    images = Train.images;
    labels = Train.labels;
    testImages = Test.testImages;
    testLabels = Test.testLabels;
    opts.imageDim = Info.imageDim;
    opts.imageChannel = Info.imageChannel;
    opts.numClasses = Info.numClasses;
    cnn.layers = {
        struct('type', 'c', 'numFilters', 6, 'filterDim', 5, 'activation_function', 'sigmoid') 
        struct('type', 'p', 'poolDim', 2, 'numfuzzypartition', 3, 'attentionkernel', 5) 
%         struct('type', 'c', 'numFilters', 8, 'filterDim', 5, 'activation_function', 'sigmoid') 
%         struct('type', 'p', 'poolDim', 2, 'numfuzzypartition', 3, 'attentionkernel', 5) 
        };
    cnn = InitializeParameters(cnn,opts);
    [Test_accmat, Train_accmat, Train_time, Test_time, Cost, cnn] = cnnTrain(cnn,images,labels,testImages,testLabels,Info);
    file = fullfile('./../../result', 'FuConvNet', [image_names{1, i} '_result.mat']);
    save (file, 'Test_accmat', 'Train_accmat', 'Train_time', 'Test_time', 'Cost', 'cnn');
    clearvars -except opts i image_info image_dir image_names  image_numel;
end