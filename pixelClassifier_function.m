function imL = pixelClassifier_function(I, env, model, ID, mapinfo)
% for running pixelClassifier in blockproc.
%
% Inputs:       I           image  block from input (marix nxmxp, where p
%                           is the number of predictor variables)
%               env         environment variables
%               model       Model structure with the necessary data
%               ID          String giving UAVSAR ID for one of the filters
%               mapinfo     Structure returned by geotiffinfo- used for
%                           some filters
% Output:       imL         output image, (matrix nxmx1 of classes)
% 
%  

% Written by Ethan Kyzivat.

%% environment
if env.blockProcessing == true % only run imClassify in parallel chunks if I'm not already using parallel chunks at a lower level
    nSubsets = 1;
else
    nSubsets = env.pixelClassifier.run.nSubsets;
end
R=mapinfo.SpatialRef;

%% proceed
nBands=size(I, 3);

    % set NoData values to NaN
I(repmat(any(I(:, :, env.radar_bands)<0,3), [1,1, size(I,3)]))=NaN; % set pixels to NaN if any radar band <0 HERE TODO: note that if import type doesn't have attached inc band (like LUT-Fr), this method fails to ID NoData areas...  
I(repmat(all(I(:, :,env.radar_bands)==env.constants.noDataValue, 3), [1,1, size(I,3)]))=NaN; % set pixels to NaN if each radar band ==0

        % remove NaN's
% % Note section under heading 'mask out near range, if applicable' adds
% NaNs back in...
%     I(repmat(isnan(I(:,:,nBands)),...
%         [1, 1, nBands]))=env.constants.noDataValue;

%% mask out near range, if applicable
if ~isnan(env.inc_band) & (env.IncMaskMin> 0 || env.IncMaskMax < Inf) % if input type doesn't use inc as feature, mask out near and far range inc angles bc they are unreliable
    if size(I, 3) <4 % no inc band was included
        error('No inc. band found?')
    else % inc band was included
%         fprintf('Masking out inc. angle < %0.2f and > %0.2f.\n', env.IncMaskMin, env.IncMaskMax)
        msk=I(:,:,env.inc_band) < env.IncMaskMin |...
            I(:,:,env.inc_band) > env.IncMaskMax; % negative mask for near/far range
        I(repmat(msk, [1,1, nBands]))=NaN;  % BW=logical(repmat(BW, [1 1 3]));
    end
end
%% loop over bands w/i image

F=single.empty(size(I,1),size(I,2),0); % initilize
for band=1:nBands
    if ismember(band, env.radar_bands) % for radar bands
        F = cat(3, F, imageFeatures(I(:,:,band),model.sigmas,...
            model.offsets,model.osSigma,model.radii,model.cfSigma,...
            model.logSigmas,model.sfSigmas, model.use_raw_image,...
            model.textureWindows, model.speckleFilter,ID, R, mapinfo,...
            [],[], env));
%         fprintf('Computed features from band %d of %d in image %d of %d\n', band, nBands, imIndex, nImages);

    elseif ismember(band, env.inc_band) & env.use_inc_band % for incidence angle band
        F = cat(3,F,imageFeatures(I(:,:,band),[],[],[],[],[],[],[], 1, [], [],...
        [],[],[],[],[], env));
        fprintf('Computed features from band %d of %d in image %d of %d\n', band, nBands, imIndex, nImages);
    elseif ismember(band, env.dem_band) % for DEM/hgt band
        F = cat(3, F, imageFeatures(I(:,:,band),...
            [],[],[],[],[],[],[], [],...
            [], [],...
            [], [], [], model.gradient_smooth_kernel, model.tpi_kernel, env)); 
%         fprintf('Computed features from band %d of %d in image %d of %d', band, nBands, imIndex, nImages);
    else 
%         fprintf('Not using features from band %d because it is not included in type ''%s''\n', band, env.inputType)
        continue
    end
end % end loop over bands

%% now start classifying
%     fprintf('Classifying image %d of %d:  %s...\n',imIndex,length(imagePaths), imagePaths{imIndex});
[imL,classProbs] = imClassify(F,model.treeBag,nSubsets, env);  
fprintf('|'); % for error reporting