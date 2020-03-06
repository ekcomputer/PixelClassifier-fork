function [imageList,labelList,classIndices, names] = parseLabelFolder(dirPath)
% reads images and generates class-balanced labels from annotations

files = dir(dirPath);

% how many annotated images (all that don't have 'Class' in the name)
nImages = 0;
for i = 1:length(files)
    fName = files(i).name;
    if ~files(i).isdir && ...
       ~contains(fName,'Class') && ...
       ~contains(fName,'.mat') && ...
       ~contains(fName,'.db') && ...
       fName(1) ~= '.'
        nImages = nImages+1;
        imagePaths{nImages} = [dirPath filesep fName];
    end
end

% list of class indices per image
classIndices = [];
[~,imName] = fileparts(imagePaths{1});
for i = 1:length(files)
    fName = files(i).name;
    k = strfind(fName,'Class');
    if contains(fName,imName) && ~isempty(k)
        [~,imn] = fileparts(fName);
        classIndices = [classIndices str2double(imn(k(1)+5:end))];
    end
end
nClasses = length(classIndices);

% read images/labels
imageList = cell(1,nImages);
labelList = cell(1,nImages);
clear imn
for i = 1:nImages
    [I, R] = geotiffread(imagePaths{i});
    [imp,imn{i}] = fileparts(imagePaths{i});
    
    nSamplesPerClass = zeros(1,nClasses);
    lbMaps = cell(1,nClasses);
    for j = 1:nClasses
        classJ = imread([imp filesep imn{i} sprintf('_Class%02d.tif',classIndices(j))]);
        classJ = (classJ(:,:,1) > 0);
        nSamplesPerClass(j) = sum(classJ(:));
        lbMaps{j} = classJ;
    end
    
    [minNSamp,indMinNSamp] = min(nSamplesPerClass);
    
    L = uint8(zeros(size(I, 1), size(I, 2)));
    for j = 1:nClasses
        if j ~= indMinNSamp
            classJ = lbMaps{j} & (rand(size(classJ)) < minNSamp/nSamplesPerClass(j));
        else
            classJ = lbMaps{j};
        end
        L(classJ) = j;
    end

    imageList{i} = I;
    labelList{i} = L;
end
try
    names=parseTrainingFileNames(imn); % removes suffix
catch
    warning('parseTrainingFileNames failed.  Using default of imn')
    names=imn;
end
end