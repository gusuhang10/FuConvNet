 function [Test_accmat, Train_accmat, Train_time, Test_time, Cost, cnn] = cnnTrain(cnn,images,labels,testimages,testlabels,Info)
	dataname = Info.dataname;
    it = 0;
    %C = [];
    epochs = cnn.numepochs;
    minibatch = cnn.minibatch;
    momIncrease = cnn.momIncrease;
    mom = cnn.mom;
    momentum = cnn.momentum;
    Alpha = cnn.alpha;
    lambda = cnn.lambda;
    
    m = length(labels);
    numLayers = numel(cnn.layers);
    Cost = cell(epochs,1);
    [Cost{1:epochs,1}] = deal([]);
    Train_accmat = zeros(1, epochs);
    Test_accmat = zeros(1, epochs);
    Train_time = zeros(1, epochs);
    Test_time = zeros(1, epochs);
    for e = 1:epochs
        if 70 < e
            alpha = Alpha;
        elseif e < 100
            alpha = Alpha * 0.5;
        elseif e <= 200
            alpha = Alpha * 0.5 * 0.5;
        end
        % randomly permute indices of data for quick minibatch sampling
        rp = randperm(m);
        tic;
        for s=1:minibatch:(m-minibatch+1)
            %profile on;
            it = it + 1;
            % momentum enable
            if it == momIncrease
                mom = momentum;
            end

            % mini-batch pick
            mb_images = images(:,:,:,rp(s:s+minibatch-1));
            mb_labels = labels(rp(s:s+minibatch-1));

            numImages = size(mb_images,4);
            
            %feedforward            
            %convolve and pooling
            activations = mb_images;
            for l = 1:numLayers
                layer = cnn.layers{l};
                if(strcmp(layer.type,'c'))%convolutional layer
                    [activations,activations1,convolvedFeatures_ReLu] = fuzzy_convol(activations,layer.W,layer.b,layer.numfuzzypartition_conv,layer.activation_function); 
                    layer.activations1 = activations1;
                    layer.convolvedFeatures_ReLu = convolvedFeatures_ReLu;
                else
                    activations = cnnFuzzyPool(layer.poolDim,activations,layer.numfuzzypartition,layer.attentionkernel);    
                end
                layer.activations = activations;
                cnn.layers{l} = layer;
            end
            %softmax
            activations = reshape(activations,[],numImages);
            probs = exp(bsxfun(@plus, cnn.Wd * activations, cnn.bd));
            sumProbs = sum(probs, 1);
            probs = bsxfun(@times, probs, 1 ./ sumProbs); 
            %[~, preds] = max(probs,[],1);
            %preds = preds';
            %acc = sum(preds == mb_labels) / length(mb_labels);
            %% --------- Calculate Cost ----------
            logp = log(probs);
            index = sub2ind(size(logp),mb_labels',1:size(probs,2));
            ceCost = -sum(logp(index));
            wCost = 0;
            for l = 1:numLayers
                layer = cnn.layers{l};
                if(strcmp(layer.type,'c'))
                    wCost = wCost + sum(layer.W(:) .^ 2);
                end
            end
            wCost = lambda/2*(wCost + sum(cnn.Wd(:) .^ 2));
            cost = ceCost+wCost; %ignore weight cost for now     
            %Backpropagation
            %errors
            
            %softmax layer
            output = zeros(size(probs));
            output(index) = 1;
            DeltaSoftmax = (probs - output); %softmax层的Delta
            %t = -DeltaSoftmax;
            
            
            %last pooling layer
            numFilters2 = cnn.layers{numLayers-1}.numFilters;
            outputDim = size(cnn.layers{numLayers}.activations,1);
            cnn.layers{numLayers}.delta = reshape(cnn.Wd' * DeltaSoftmax,outputDim,outputDim,numFilters2,numImages);
             
            
            %other layers
            for l = numLayers-1:-1:1
                layer = cnn.layers{l};
                if(strcmp(layer.type,'c'))% convolutional layer
                    numFilters = cnn.layers{l}.numFilters;
                    outputDim = size(cnn.layers{l+1}.activations,1);                    
                    poolDim = cnn.layers{l+1}.poolDim;
                    convDim = outputDim * poolDim;
                    DeltaPool = cnn.layers{l+1}.delta; 

                    %unpool from last layer
                    DeltaUnpool = zeros(convDim,convDim,numFilters,numImages);        
                    for imNum = 1:numImages
                        for FilterNum = 1:numFilters
                            unpool = DeltaPool(:,:,FilterNum,imNum);
                            DeltaUnpool(:,:,FilterNum,imNum) = kron(unpool,ones(poolDim)); 
                        end
                    end
                    if (strcmp(layer.activation_function,'ReLU'))
                        convolvedFeatures_ReLu = layer.convolvedFeatures_ReLu;
                        DeltaConv = DeltaUnpool .* convolvedFeatures_ReLu;
                    elseif (strcmp(layer.activation_function,'sigmoid'))
                        activations = layer.activations;
                        DeltaConv = DeltaUnpool .* activations .* (1-activations);
                    end
                    layer.delta = DeltaConv;
                    cnn.layers{l} = layer;
                else
                    numFilters1 = cnn.layers{l-1}.numFilters;
                    numFilters2 = cnn.layers{l+1}.numFilters;
                    outputDim1 = size(layer.activations,1);
                    DeltaPooled = zeros(outputDim1,outputDim1,numFilters1,numImages);
                    DeltaConv = cnn.layers{l+1}.delta;
                    Wc = cnn.layers{l+1}.W;
                    for i = 1:numImages
                        for f1 = 1:numFilters1
                            for f2 = 1:numFilters2 
                                DeltaPooled(:,:,f1,i) = DeltaPooled(:,:,f1,i) + convn(DeltaConv(:,:,f2,i),Wc(:,:,f1,f2),'full');
                            end
                        end
                    end
                    layer.delta = DeltaPooled;
                    cnn.layers{l} = layer;
                end
            end
            
            %gradients
            activationsPooled = cnn.layers{numLayers}.activations;
            activationsPooled = reshape(activationsPooled,[],numImages);
            Wd_grad = DeltaSoftmax*(activationsPooled)';
            bd_grad = sum(DeltaSoftmax,2);
            
            cnn.Wd_velocity = mom*cnn.Wd_velocity + alpha * (Wd_grad/minibatch+lambda*cnn.Wd);
            cnn.bd_velocity = mom*cnn.bd_velocity + alpha * (bd_grad/minibatch);
            cnn.Wd = cnn.Wd - cnn.Wd_velocity;
            cnn.bd = cnn.bd - cnn.bd_velocity;
            
            %other convolutions
            for l = numLayers:-1:1
                layer = cnn.layers{l};
                if(strcmp(layer.type,'c'))%if this is a convolutional layer
                    numFilters2 = layer.numFilters;
                    if(l == 1)
                        numFilters1 = cnn.imageChannel;
                        %activationsPooled = mb_images;%%%%
                        activationsPooled = cnn.layers{l}.activations1;
                    else
                        numFilters1 = cnn.layers{l-2}.numFilters;
                        %activationsPooled = cnn.layers{l-1}.activations;%%%%%
                        activationsPooled = cnn.layers{l}.activations1;
                    end
                    Wc_grad = zeros(size(layer.W));
                    bc_grad = zeros(size(layer.b));
                    DeltaConv = layer.delta;
%
                    [C1,C2,~,~] = size(DeltaConv);
                    DeltaConv2 = permute(DeltaConv, [2 1 3 4]);
                    activationsPooled = permute(activationsPooled, [2 1 3 4]);
                    for fil2 = 1:numFilters2
                        deltaconv = DeltaConv2(:,:,fil2,:);
                        deltaconv = reshape(deltaconv, [C1*C2 1 1 numImages]);
                        deltaconv = repmat(deltaconv, [1 1 numFilters1 1]);
                        Wt = pagemtimes(activationsPooled, deltaconv);
                        Wt = sum(Wt, 4);
                        Wt = reshape(Wt, layer.filterDim, layer.filterDim, numFilters1);
                        Wt = permute(Wt, [2 1 3]);
                        Wc_grad(:,:,:,fil2) = Wt;
                        temp = DeltaConv(:,:,fil2,:);
                        bc_grad(fil2) = sum(temp(:));
                    end

                    layer.W_velocity = mom*layer.W_velocity + alpha*(Wc_grad/numImages + lambda*layer.W); 
                    layer.b_velocity = mom*layer.b_velocity + alpha*(bc_grad/numImages);
                    layer.W = layer.W - layer.W_velocity;
                    layer.b = layer.b - layer.b_velocity;                   
                end 
                cnn.layers{l} = layer;
            end
            fprintf('%s:Epoch %d: Cost on iteration %d is %f\n', dataname, e, it, cost);
            Cost{e}(length(Cost{e})+1) = cost;
            %break;
        end
        Train_time(e) = toc;
        Train_accmat(e) = cnnTest(cnn,images,labels);
        fprintf('%s:Train Accuracy is %f\n', dataname, Train_accmat(e));
        tic;
        Test_accmat(e) = cnnTest(cnn,testimages,testlabels);
        Test_time(e) = toc;
        fprintf('%s:Test Accuracy is %f\n', dataname, Test_accmat(e));
    end
 end
 