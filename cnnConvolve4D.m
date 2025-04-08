function [convolvedFeatures, convolvedFeatures_ReLu] = cnnConvolve4D(images,W,b,activation_function)
    filterDim = size(W,1);
    numFilters1 = size(W,3);
    numFilters2 = size(W,4);
    numImages = size(images,4);
    imageDim = size(images,1);
    convDim = imageDim-filterDim+1;
    convolvedImage_ReLU = zeros(convDim,convDim);
    convolvedFeatures = zeros(convDim,convDim,numFilters2,numImages);
    for i = 1:numImages
        for fil2 = 1:numFilters2
            convolvedImage = zeros(convDim, convDim);
            for fil1 = 1:numFilters1
                filter = squeeze(W(:,:,fil1,fil2));
                filter = rot90(squeeze(filter),2);
                im = squeeze(images(:,:,fil1,i));
                convolvedImage = convolvedImage + conv2(im,filter,'valid');
            end
            convolvedImage = bsxfun(@plus,convolvedImage,b(fil2));
            if activation_function == "ReLU"
                convolvedImage = max(convolvedImage_ReLU, convolvedImage);
            elseif activation_function == "sigmoid"
                convolvedImage = 1 ./ (1+exp(-convolvedImage));
            end
            convolvedFeatures(:, :, fil2, i) = convolvedImage;
        end
    end
    convolvedFeatures_ReLu = ones(size(convolvedFeatures));
    convolvedFeatures_ReLu(convolvedImage<=0) = 0;
end