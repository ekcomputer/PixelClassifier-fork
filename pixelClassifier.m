clear, clc
tic
%% set parameters
Env_PixelClassifier % load environment vars
testPath = env.output.test_dir;
% where images are

outputMasks = env.pixelClassifier.run.outputMasks;
% if to output binary masks corresponding to pixel classes

outputProbMaps = env.pixelClassifier.run.outputProbMaps; %true;
% if to output probability maps from which output masks are derived

modelPath = env.output.current_model;
% where the model is

nSubsets = env.pixelClassifier.run.nSubsets;
% the set of pixels to be classified is split in this many subsets;
% if nSubsets > 1, the subsets are classified using 'parfor' with
% the currently-opened parallel pool (or a new default one if none is open);
% see imClassify.m for details;
% it's recommended to set nSubsets > the number of cores in the parallel pool;
% this can make classification substantially faster than when a
% single thread is used (nSubsets = 1).  Divides image into nSubsets
% parts to classify, so numel(F)/nSubsets should fit into memory

% 
% no parameters to set beyond this point
%

%% load image paths, model

disp('loading model')
tic
load(modelPath); % loads model
toc

files = dir(testPath);
nImages = 0;
for i = 1:length(files)
    fName = files(i).name;
    if ~contains(fName,'.db') && ~contains(fName,'cls') && ~contains(fName,'Class') && fName(1) ~= '.' && endsWith(fName,'.tif')
        nImages = nImages+1;
        imagePaths{nImages} = [testPath filesep fName];
    end
end

%% classify

for imIndex = 1:length(imagePaths)
    I = imreadGrayscaleDouble(imagePaths{imIndex});
    nBands=size(I, 3);
    tic
    for band=1:nBands
        fprintf('computing features from band %d of %d in image %d of %d\n', band, nBands, imIndex, nImages);
        F0 = imageFeatures(I(:,:,band),model.sigmas,model.offsets,model.osSigma,model.radii,model.cfSigma,model.logSigmas,model.sfSigmas, model.use_raw_image, model.textureWindows);
        if band==1
            F=F0;
        else
            F=cat(3,F,F0);
        end
    end
    fprintf('classifying image %d of %d...',imIndex,length(imagePaths));
    [imL,classProbs] = imClassify(F,model.treeBag,nSubsets);
    fprintf('time: %f s\n', toc);

    [fpath,fname] = fileparts(imagePaths{imIndex});
    for pmIndex = 1:size(classProbs,3)
        if outputMasks
            imwrite(imL == pmIndex,[fpath filesep fname sprintf('_Class%d.png',pmIndex)]);
        end
        if outputProbMaps
            imwrite(classProbs(:,:,pmIndex),[fpath filesep fname sprintf('_Class%d_PM.png',pmIndex)]);
        end
    end
end

disp('done classifying')
toc
%% combine images
for n=1:length(imagePaths)
    [~, g(n).basename, ~]=fileparts(imagePaths{n});
%     g(n).basename=[g(n).basename, '.tif'];
    addOutputImages(g(n).basename);
end

fprintf('Combined output images to: %s\n', env.output.test_dir)