clear
% clc
fprintf('\n\n')
%% set parameters
Env_PixelClassifier % load environment vars
trainPath = env.output.train_dir;
% where images and labes are;
% images are assumed to have the same size;
% every image should have the same number of accompanied label masks,
% labeled <image name>_ClassX.png, where X is the index of the label;
% labels can be created using ImageAnnotationBot:
% https://www.mathworks.com/matlabcentral/fileexchange/64719-imageannotationbot

sigmas = env.pixelClassifier.sigmas; %[1 2 3];
% basic image features are simply derivatives (up to second order) in different scales;
% this parameter specifies such scales; details in imageFeatures.m

offsets = env.pixelClassifier.offsets; %[3 5]; %OPTIONAL
% in pixels; for offset features (see imageFeatures.m)
% set to [] to ignore offset features
osSigma = env.pixelClassifier.osSigma; %2;
% sigma for offset features

radii = env.pixelClassifier.radii;%[15 20 25]; %OPTIONAL
% range of radii on which to compute circularity features (see imageFeatures.m)
% set to [] to ignore circularity features
cfSigma = env.pixelClassifier.cfSigma; %2;
% sigma for circularity features

logSigmas = env.pixelClassifier.logSigmas;%[1 2]; %OPTIONAL
% sigmas for LoG features (see imageFeatures.m)
% set to [] to ignore LoG features

sfSigmas = env.pixelClassifier.sfSigmas;%[1 2]; %OPTIONAL
% steerable filter features sigmas (see imageFeatures.m)
% set to [] to ignore steerable filter features

nTrees = env.pixelClassifier.nTrees; %20;
% number of decision trees in the random forest ensemble

minLeafSize = env.pixelClassifier.minLeafSize; %60;
% minimum number of observations per tree leaf

pctMaxNPixelsPerLabel = env.pixelClassifier.pctMaxNPixelsPerLabel;
% percentage of max number of pixels per label (w.r.t. num of pixels in image);
% this puts a cap on the number of training samples and can improve training speed

modelPath = env.output.current_model;
% path to where model will be saved



use_raw_image=env.pixelClassifier.use_raw_image;
textureWindows=env.pixelClassifier.textureWindows;
speckleFilter=env.pixelClassifier.speckleFilter;

% 
% no parameters to set beyond this point
%
%% read images/labels

[imageList,labelList,labels, names, maprefs, mapinfos] = parseLabelFolder(trainPath);
nLabels = length(labels);
nImages = length(imageList);

%% remove image no data masks
for imageIndex=1:nImages
    sImage=size(imageList{imageIndex}, 3);
    imageList{imageIndex}(repmat(isnan(imageList{imageIndex}(:,:,sImage)), [1, 1, sImage]))=env.constants.noDataValue;
end
%% training samples cap

maxNPixelsPerLabel = (pctMaxNPixelsPerLabel/100)*size(imageList{1},1)*size(imageList{1},2);
for imIndex = 1:nImages
    L = labelList{imIndex};
    for labelIndex = 1:nLabels
        LLI = L == labelIndex;
        nPixels = sum(sum(LLI));
        rI = rand(size(L)) < maxNPixelsPerLabel/nPixels;
        L(LLI) = 0;
        LLI2 = rI & (LLI > 0);
        L(LLI2) = labelIndex;
    end
    labelList{imIndex} = L;
end

%% construct train matrix


tic
nBands=size(imageList{1}, 3);
lb_all=[];
ft_all=[]; %double.empty(0,nBands); % initilize
for imIndex = 1:nImages % loop over images 
    L = labelList{imIndex}; labelList{imIndex}=[]; % save mem
    ft_band=[];
%     training(band).lb = [];
    for band=1:nBands % loop over bands w/i image
        if nBands~=size(imageList{imIndex}, 3)
            txt=sprintf('EK: Number of bands in image %d is %d, while it is %d in image 1.', band,size(imageList{imIndex}, 3), size(imageList{1}, 3));
            error(txt);
        end
        training(band).ft = [];
        fprintf('computing features from band %d of %d in image %d of %d\n', band, nBands, imIndex, nImages);
        if band~=nBands && (strcmp(env.inputType, 'Freeman-inc') || strcmp(env.inputType, 'C3-inc') || strcmp(env.inputType, 'Norm-Fr-C11-inc') )
            [F,featNames] = imageFeatures(imageList{imIndex}(:,:,band),sigmas,offsets,osSigma,radii,cfSigma,logSigmas,sfSigmas, use_raw_image, textureWindows, speckleFilter, names{imIndex}, maprefs{imIndex}, mapinfos{imIndex});
        else % last band is range band- only use raw image
                % here, F gets rewritten for each band
            [F,featNames_last_band] = imageFeatures(imageList{imIndex}(:,:,band),[],[],[],[],[],[],[], 1, [], []);
            featNames(end+1)=featNames_last_band;
        end
%         if band==1 && strcmp(env.inputType, 'Freeman-inc') % only compute labels for first band of image
%             [rfFeat,lb] = rfFeatAndLab(F,L);            
%         else
%             rfFeat = rfFeatAndLab(F, L);
%         end
        [rfFeat,lb_band] = rfFeatAndLab(F,L);
        ft_band = [ft_band, rfFeat]; % can just say training(band).ft = rfFeat; ...
    end
%     lb_all = [lb_all; rfLbl];
    lb_all=[lb_all; lb_band];
    ft_all=[ft_all; ft_band];
    imageList{imIndex}=[]; % clear to save memory
end
% clear F % save memory
fprintf('time spent computing features: %f s\n', toc);

%% concat training matrices (one for each band)
    % save original total features and labels before partitioning
% ft_all=[training.ft];
% lb=[training.lb];
% lb_all=lb; clear lb; 
%% split into training and val datasets; turn labels into categories
    % lb and ft are training partitions, lb_val and ft_val are validation
    % partitions, lb_all and ft_all include both
global env
rng(env.seed);
c = cvpartition(lb_all,'Holdout',env.valPartitionRatio);
for p=1:length(lb_all)
    lb_all_cell{p}=sprintf('%02d-%s',lb_all(p), env.class_names{lb_all(p)});
end
ft=ft_all(c.training(1),:);
lb=lb_all_cell(c.training(1));
ft_subset_validation=ft_all(c.test(1),:);
lb_subset_validation=lb_all_cell(c.test(1));
%% training

fprintf('training...'); tic
% rng('shuffle')
[treeBag,featImp,oobPredError] = rfTrain(ft,lb,nTrees,minLeafSize, env.seed);
figureQSS
subplot(1,2,1), 
if strcmp(env.inputType, 'Freeman-inc')
%     featImp=[featImp, zeros(1, length(featNames)*nBands-length(featImp))]; 
    featImp=[featImp, zeros(1, length(featNames)-2)]; 
elseif strcmp(env.inputType, 'C3-inc')
%     featImp=[featImp, zeros(1, length(featNames)*nBands-length(featImp))]; 
    featImp=[featImp, zeros(1, length(featNames)-2)]; %%HERE TODO
else
end
featImpRshp=reshape(featImp, [length(featImp)/nBands, nBands ]); %% <----HERE
barh(featImpRshp), set(gca,'yticklabel',featNames'), set(gca,'YTick',1:length(featNames)), title('feature importance')
legend_txt=env.plot.bandLabels;
% legend_txt=cellstr(num2str([1:nBands]'));
legend(legend_txt, 'Location', 'best', 'FontSize', 12);
subplot(1,2,2), plot(oobPredError), title('out-of-bag classification error')
fprintf('training time: %f s\n', toc);

%% validation: confusion matrix on only test subset of image
    % reconstruct F
% F=cat(3, F{1}, F{2}, F{3});
% imL = imClassify(F,treeBag,1);
[~,scores] = predict(treeBag,ft_subset_validation); % can use ft_all, but that might be cheating; ft_val is a k-fold subset
[~,lb_val_test] = max(scores,[],2);
for p=1:length(lb_val_test)
    lb_val_test_cell{p}=sprintf('%02d-%s',lb_val_test(p), env.class_names{lb_val_test(p)});
end
[v.C, v.cm, v.order, v.k, v.OA]=confusionmatStats(lb_subset_validation,lb_val_test_cell, env.class_names); %% <======= HERE 1/9
%% save model

model.treeBag = treeBag;
model.sigmas = sigmas;
model.offsets = offsets;
model.osSigma = osSigma;
model.radii = radii;
model.cfSigma = cfSigma;
model.logSigmas = logSigmas;
model.sfSigmas = sfSigmas;
model.oobPredError=oobPredError;
model.featImp=featImp;
model.featNames=featNames;
model.use_raw_image=use_raw_image;
model.textureWindows=textureWindows;
model.speckleFilter=speckleFilter;
model.env=env;
model.validation=v;
save(modelPath,'model');

disp('done training')

%% classify, without having to click again
drawnow
% pixelClassifier