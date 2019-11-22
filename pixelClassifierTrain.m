clear, clc

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


% 
% no parameters to set beyond this point
%
use_raw_image=env.pixelClassifier.use_raw_image;
textureWindows=env.pixelClassifier.textureWindows;
%% read images/labels

[imageList,labelList,labels] = parseLabelFolder(trainPath);
nLabels = length(labels);

%% training samples cap

maxNPixelsPerLabel = (pctMaxNPixelsPerLabel/100)*size(imageList{1},1)*size(imageList{1},2);
nImages = length(imageList);
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
for imIndex = 1:nImages % loop over images 
    L = labelList{imIndex};
    lb=[];
%     training(band).lb = [];
    for band=1:nBands % loop over bands w/i image
        if nBands~=size(imageList{imIndex}, 3)
            txt=sprintf('Number of bands in image %d is %d, while it is %d in image 1.', band,size(imageList{imIndex}, 3), size(imageList{1}, 3));
            error(txt);
        end
        training(band).ft = [];
        fprintf('computing features from band %d of %d in image %d of %d\n', band, nBands, imIndex, nImages);
        [F,featNames] = imageFeatures(imageList{imIndex}(:,:,band),sigmas,offsets,osSigma,radii,cfSigma,logSigmas,sfSigmas, use_raw_image, textureWindows);
        if band==1 % only compute labels for first band of image
            [rfFeat,rfLbl] = rfFeatAndLab(F,L);
        else
            rfFeat = rfFeatAndLab(F, L);
        end
        training(band).ft = [training(band).ft; rfFeat];
    end
    lb = [lb; rfLbl];
end
% clear F % save memory
fprintf('time spent computing features: %f s\n', toc);

%% concat training matrices (one for each band)
ft=[training.ft];
% lb=[training.lb];

%% training

fprintf('training...'); tic
[treeBag,featImp,oobPredError] = rfTrain(ft,lb,nTrees,minLeafSize);
figureQSS
subplot(1,2,1), barh(featImp), set(gca,'yticklabel',featNames'), set(gca,'YTick',1:length(featNames)), title('feature importance')
subplot(1,2,2), plot(oobPredError), title('out-of-bag classification error')
fprintf('training time: %f s\n', toc);

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
model.env=env;
save(modelPath,'model');

disp('done training')

%% classify, without having to click again
pixelClassifier