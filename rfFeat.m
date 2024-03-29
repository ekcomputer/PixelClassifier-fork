function [rfFeat] = rfFeat(imFeat,imLbl)
% extracts just features as vector from set of image/labe-image pairs
nVariables = size(imFeat,3);

nLabels = max(max(imLbl)); % assuming labels are 1, 2, 3, ...

nPixelsPerLabel = zeros(1,nLabels);
pxlIndices = cell(1,nLabels);

for i = 1:nLabels
    pxlIndices{i} = find(imLbl == i);
    nPixelsPerLabel(i) = numel(pxlIndices{i});
end

nSamples = sum(nPixelsPerLabel);

rfFeat = zeros(nSamples,nVariables);

offset = [0 cumsum(nPixelsPerLabel)];
for i = 1:nVariables
    F = imFeat(:,:,i);
    for j = 1:nLabels
        rfFeat(offset(j)+1:offset(j+1),i) = F(pxlIndices{j});
    end
end


end