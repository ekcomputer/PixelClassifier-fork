clear, clc

%% set parameters
Env_PixelClassifier % load environment vars
testPath = env.output.test_dir;
% where images are

outputMasks = true;
% if to output binary masks corresponding to pixel classes

outputProbMaps = false; %true;
% if to output probability maps from which output masks are derived

modelPath = env.output.current_model;
% where the model is

nSubsets = 100;
% the set of pixels to be classified is split in this many subsets;
% if nSubsets > 1, the subsets are classified using 'parfor' with
% the currently-opened parallel pool (or a new default one if none is open);
% see imClassify.m for details;
% it's recommended to set nSubsets > the number of cores in the parallel pool;
% this can make classification substantially faster than when a
% single thread is used (nSubsets = 1).

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
    if ~contains(fName,'.db') && ~contains(fName,'Class') && fName(1) ~= '.'
        nImages = nImages+1;
        imagePaths{nImages} = [testPath filesep fName];
    end
end

%% classify

for imIndex = 1:length(imagePaths)
    fprintf('classifying image %d of %d...',imIndex,length(imagePaths));
    I = imreadGrayscaleDouble(imagePaths{imIndex});

    tic
    F = imageFeatures(I,model.sigmas,model.offsets,model.osSigma,model.radii,model.cfSigma,model.logSigmas,model.sfSigmas);
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

%% combine images
for n=1:length(imagePaths)
    [~, g(n).basename, ~]=fileparts(imagePaths{n});
%     g(n).basename=[g(n).basename, '.tif'];
    addOutputImages(g(n).basename);
end