function [rfFeat,rfLbl, rfInfo] = rfFeatAndLab(imFeat,imLbl, varargin)
% extracts features and labels as vector from set of image/labe-image pairs
% Inputs:       varargin =  file number (scalar)
if ~isempty(varargin)
    im_num=varargin{1};
    
    %% imlabel to get indiv polygons from label masks
    L=bwlabel(imLbl);
else
    im_num=0;
    L=zeros(size(imLbl));
end

nVariables = size(imFeat,3);

nLabels = max(max(imLbl)); % assuming labels are 1, 2, 3, ...

nPixelsPerLabel = zeros(1,nLabels);
pxlIndices = cell(1,nLabels);

for i = 1:nLabels
    pxlIndices{i} = find(imLbl == i); % pixel idx for each class
    nPixelsPerLabel(i) = numel(pxlIndices{i}); % vector: number of px for each class
end

nSamples = sum(nPixelsPerLabel); % scalar: total number of px for all classes

rfFeat = zeros(nSamples,nVariables);
rfLbl = zeros(nSamples,1);
rfInfo = zeros(nSamples,2, 'uint16'); % column one is image number, columne 2 is polygon/region number

offset = [0 cumsum(nPixelsPerLabel)];
for i = 1:nVariables
    F = imFeat(:,:,i); % feature i
    for j = 1:nLabels
        rfFeat(offset(j)+1:offset(j+1),i) = F(pxlIndices{j});
    end
end
for j = 1:nLabels
    rfLbl(offset(j)+1:offset(j+1)) = j;
    rfInfo(offset(j)+1:offset(j+1),1) = im_num; % stamp in file number
    rfInfo(offset(j)+1:offset(j+1),2) = L(pxlIndices{j}); % stamp in polygon number
end

end