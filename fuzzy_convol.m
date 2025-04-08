function [convolvedFeatures,wt,convolvedFeatures_ReLu] = fuzzy_convol(images,W,b,numfuzzypartition,activation_function)

filterDim = size(W,1);
numFilters1 = size(W,3);
numFilters2 = size(W,4);
numImages = size(images,4);
imageDim = size(images,1);
convDim = imageDim-filterDim+1;
R = filterDim^2;
RuleNum = filterDim^2;%
[~,~,channel_data,~] = size(images);
[ker_1,ker_2,~,~] = size(W);
%WT_ReLU = zeros(convDim*convDim, ker_1*ker_2, numFilters1);
%convolvedFeatures = zeros(convDim,convDim,numFilters2,numImages);
convolvedImage = zeros(convDim,convDim,numFilters2,numImages);
wt = zeros(convDim*convDim, filterDim*filterDim, numFilters1, numImages);
RuleBunch = zeros(RuleNum,R,numFilters1);
convolvedFeatures_ReLu = ones(convDim,convDim,numFilters2,numImages);
fuzzy_center = inimemfun_input([],numfuzzypartition(1));

W_Row = permute(W, [2 1 3 4]);
W_Row = reshape(W_Row, ker_1*ker_2, 1, numFilters1, numFilters2);
if numImages <= 1000
    RuleBunch_list_num = floor(numImages/4) * RuleNum;
else
    RuleBunch_list_num = 500 * RuleNum;
end
[RuleBunch_list,~] = GetRuleBunch(fuzzy_center,RuleBunch_list_num,RuleNum,[]);
for i = 1:numImages
    for fil1 = 1:numFilters1
        rand_num = randperm(RuleBunch_list_num, R);
        RuleBunch(:,:,fil1) = RuleBunch_list(rand_num, :);
    end
    image = images(:,:,:,i);

    images_Row = zeros(ker_1*ker_2, convDim*convDim, channel_data);
    image = permute(image, [2 1 3]);
    for j = 1:numFilters1
        image_Row = im2col(image(:,:,j), [ker_2 ker_1], 'sliding');
        images_Row(:,:,j) = image_Row;
    end
    datamat = permute(images_Row, [2 1 3]);
    RuleBunch2 = permute(RuleBunch, [2 1 3]);
    WT = pagemtimes(datamat, RuleBunch2);
    WT = 1./ (1+exp(-WT));
    WT1 = sum(WT,2);
    WT = WT ./ WT1;
    WT = 0.01 * WT + datamat;
    result = pagemtimes(WT, W_Row);
    Result = sum(result, 3);
    Result = reshape(Result, [convDim convDim numFilters2]);
    convolvedimage = permute(Result, [2 1 3]);
    convolvedImage(:,:,:,i) = convolvedimage;
    wt(:,:,:,i) = WT;
end
B = zeros(1,1,numFilters2,numImages);
for t = 1:numFilters2
    B(:,:,t, :) = b(t);
end
convolvedImage = convolvedImage + B;
if activation_function == "ReLU"
    RELU_grad = ones(size(convolvedImage));
    RELU_grad(convolvedImage<=0) = 0;
    convolvedFeatures = max(zeros(size(convolvedImage)),convolvedImage);
    convolvedFeatures_ReLu = RELU_grad;
elseif activation_function == "sigmoid"
    convolvedFeatures = 1 ./ (1+exp(-convolvedImage));
end
end