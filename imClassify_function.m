function imL = imClassify_function(rfFeat,treeBag) % [imL,classProbs]
% for blockproc version of imClassify. Inputs: matrix of image features, model,
% fraction of image to process in each tile
%
% Outputs: classified image (and class probs) of same rows and cols as
% image in mat-file path (and with as many bands as features, if using)
%
% Unused bc blockproc can't use .mat file as input...crazy!

global env

%% extract block
rfFeat=rfFeat.data;

%% reshape rfFeat from no.entries x no. features to linear
[nr,nc,nVariables] = size(rfFeat);
rfFeat = reshape(rfFeat,[nr*nc,nVariables]);

%% branch for parallel
 
% pre-allocate answer
scores=zeros([size(rfFeat, 1), length(treeBag.ClassNames)], 'like', rfFeat);
indOfMax=uint8(env.constants.noDataValue_ouput)*ones([size(rfFeat, 1),1], 'uint8'); % same as zeros

    % generate mask
msk=any(isnan(rfFeat),2); % negative mask

    % predict only on valid data
[~,scores_valid] = predict(treeBag,rfFeat(~msk,:));
[~,indOfMax_valid] = max(scores_valid,[],2);

    % add in classified valid data to pre-allocated zeros, using mask

scores(~msk,:)=scores_valid;  
indOfMax(~msk)=indOfMax_valid;

%% Reshape outputs into image ( could be slightly faster if vectorized, but only takes 3 sec)
imL = reshape(indOfMax,[nr,nc]); % already uint8
% classProbs = zeros(nr,nc,size(scores,2), 'single');
% for i = 1:size(scores,2)
%     classProbs(:,:,i) = reshape(scores(:,i),[nr,nc]);