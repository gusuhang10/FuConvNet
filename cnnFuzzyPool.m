function pooledFeatures = cnnFuzzyPool(poolDim, convolvedFeatures, numfuzzypartion,attentionkernel)

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);
pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);
[c,sig] = inimemfun([], convolvedDim, convolvedDim, numfuzzypartion);
featuremap_fuzzy = cell(1,numFilters);
featuremap_mean_pool = zeros(1,numFilters);

    for imageNum = 1:numImages
        fuzzysum = 0;
        for featureNum = 1:numFilters
            featuremap = squeeze(convolvedFeatures(:,:,featureNum,imageNum));
            featuremap_fuzzy{featureNum} = calmemfun(featuremap, c, sig, numfuzzypartion, 1);
            gg = zeros(convolvedDim, convolvedDim);
            for i = 1:numfuzzypartion
                %if sum((gg), [1 2]) < sum(featuremap_fuzzy{featureNum}{i}, [1 2]) 
                if sum(sum((gg))) < sum(sum(featuremap_fuzzy{featureNum}{i})) 
                    gg = featuremap_fuzzy{featureNum}{i};
                end
            end
            featuremap_mean_pool(featureNum) = sum(sum(gg));
            fuzzysum = fuzzysum + featuremap_mean_pool(featureNum);
        end
        channel_w1 = featuremap_mean_pool / fuzzysum;
        channel_w2 = 1 - channel_w1;
        channel_w3 = channel_w1 .* channel_w2;
        channel_w = channel_w3 ./ sum(channel_w3);

        gg_t = zeros(convolvedDim, convolvedDim);
        for featureNum = 1:numFilters
            convolvedFeatures(:,:,featureNum,imageNum) = channel_w(featureNum) * convolvedFeatures(:,:,featureNum,imageNum);
            featuremap = squeeze(convolvedFeatures(:,:,featureNum,imageNum));
            featuremap_fuzzy{featureNum} = calmemfun(featuremap, c, sig, numfuzzypartion, 1);
            gg = zeros(convolvedDim, convolvedDim);            
            for i = 1:numfuzzypartion
                %if sum((gg), [1 2]) < sum(featuremap_fuzzy{featureNum}{i}, [1 2]) 
                if sum(sum((gg))) < sum(sum(featuremap_fuzzy{featureNum}{i})) 
                    gg = featuremap_fuzzy{featureNum}{i};
                end
            end
            gg_t = gg_t + gg;
        end

        gg_t_1 = gg_t / numFilters;
        gg_t_2 = conv2(gg_t,ones(attentionkernel)/(attentionkernel.^2),'same') / numFilters;
        for featureNum = 1:numFilters
            convolvedFeatures(:,:,featureNum,imageNum) = convolvedFeatures(:,:,featureNum,imageNum) .* gg_t_1 .* gg_t_2;
            featuremap = squeeze(convolvedFeatures(:,:,featureNum,imageNum));
            pooledFeaturemap = conv2(featuremap,ones(poolDim),'valid');
            pooledFeatures(:,:,featureNum,imageNum) = pooledFeaturemap(1:poolDim:end,1:poolDim:end);
        end
    end																	
end