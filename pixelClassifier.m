% Note "if 1==1 switch to avoid using blockproc"
clear
% clc
fprintf('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nStarting classification queue...\n')
if isunix % make sure gdal paths work properly
    unix('source /opt/PGSCplus-2.2.2/init-gdal.sh');
end
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
totalTime=tic;
load(modelPath); % loads model
toc

files = dir(testPath);
nImages = 0;
for i = 1:length(files)
    fName = files(i).name;
    if ~contains(fName,'.db') && ~contains(fName,'cls') && ~contains(fName,'Class') && fName(1) ~= '.' && endsWith(fName,'.tif')
        nImages = nImages+1;
        imagePaths{nImages} = [testPath filesep fName];
        names_raw{nImages}=fName;
    end
end
%% parse names 
try
    names=parseTrainingFileNames(names_raw); % removes suffix
catch
    warning('parseTrainingFileNames failed.  Using default of ''NoNameFound''')
    names={'NoNameFound'};
end
%% classify
% parpool(2)
for imIndex = 1:length(imagePaths)
    try % for fault tolerance so following images will still run even if an error...
            % check if image already exists...
        f.pth_out=[imagePaths{imIndex}(1:end-4), '_cls.tif'];
        if exist(f.pth_out)==2 % don't overwrite if it exists
            fprintf('File:\t%s\n\talready exists, so not overwriting.\n', f.pth_out)
            continue
        end
        fprintf('\n-------------------------\nLoading file:\t%s\n', imagePaths{imIndex})
        try
            mapinfo=geotiffinfo(imagePaths{imIndex});
            [I,R] = geotiffread(imagePaths{imIndex});
            if isempty(R) || isempty(mapinfo.SpatialRef)
                error('EK: Empty R')
            end
        catch w
            fprintf('EK Warning: initial geotifinfo failed. Using gdaledit and retrying...\n\tFile: %s', imagePaths{imIndex})
                
                % use gdal_edit and values from other matlab I/O functions
                % to re-assign SRS info if image is normal tif + tfw and
                % proj
            worldfile=[imagePaths{imIndex}(1:end-4), '.tfw'];
            im_info=imfinfo(imagePaths{imIndex});
            co=worldfileread(worldfile, 'planar', [im_info.Height, im_info.Width]);
            bb=[co.XWorldLimits(1), co.YWorldLimits(2), co.XWorldLimits(2), co.YWorldLimits(1)];
            cmd=sprintf('gdal_edit.py -a_srs "EPSG:102001" -a_ullr %s %s', num2str(bb), imagePaths{imIndex})
            system(cmd);
            mapinfo=geotiffinfo(imagePaths{imIndex});
            [I,R] = geotiffread(imagePaths{imIndex});
        end
        nBands=size(I, 3);

            % set NoData values to NaN
        I(repmat(any(I(:, :, env.radar_bands)<0,3), [1,1, size(I,3)]))=NaN; % set pixels to NaN if any radar band <0 HERE TODO: note that if import type doesn't have attached inc band (like LUT-Fr), this method fails to ID NoData areas...  
        I(repmat(all(I(:, :,env.radar_bands)==env.constants.noDataValue, 3), [1,1, size(I,3)]))=NaN; % set pixels to NaN if each radar band ==0

            % remove NaN's
    % % Note section under heading 'mask out near range, if applicable' adds
    % NaNs back in...
    %     I(repmat(isnan(I(:,:,nBands)),...
    %         [1, 1, nBands]))=env.constants.noDataValue;
        stepTime=tic;

        %% mask out near range, if applicable
        if ~isnan(env.inc_band) & (env.IncMaskMin> 0 || env.IncMaskMax < Inf) % if input type doesn't use inc as feature, mask out near and far range inc angles bc they are unreliable
            if size(I, 3) <4 % no inc band was included
                error('No inc. band found?')
            else % inc band was included
                fprintf('Masking out inc. angle < %0.2f and > %0.2f.\n', env.IncMaskMin, env.IncMaskMax)
                msk=I(:,:,env.inc_band) < env.IncMaskMin |...
                    I(:,:,env.inc_band) > env.IncMaskMax; % negative mask for near/far range
                I(repmat(msk, [1,1, nBands]))=NaN;  % BW=logical(repmat(BW, [1 1 3]));
            end
        end
        %% loop over bands w/i image

        F=single.empty(size(I,1),size(I,2),0); % initilize
        for band=1:nBands
            if ismember(band, env.radar_bands) % for radar bands
                F = cat(3,F,imageFeatures(I(:,:,band),model.sigmas,...
                    model.offsets,model.osSigma,model.radii,model.cfSigma,...
                    model.logSigmas,model.sfSigmas, model.use_raw_image,...
                    model.textureWindows, model.speckleFilter,...
                    names{imIndex}, R, mapinfo,...
                    [],[]));
                fprintf('Computed features from band %d of %d in image %d of %d\n', band, nBands, imIndex, nImages);
            elseif ismember(band, env.inc_band) & env.use_inc_band % for incidence angle band
                F = cat(3,F,imageFeatures(I(:,:,band),[],[],[],[],[],[],[], 1, [], [],...
                    [],[],[],[],[]));
                fprintf('Computed features from band %d of %d in image %d of %d\n', band, nBands, imIndex, nImages);
    %         elseif band == nBands && ismember(env.inputType, {'Freeman', 'C3', 'T3', 'Sinclair'})
                % Don't extract any features from inc. band.   
            elseif ismember(band, env.dem_band) % for DEM/hgt band
                F = cat(3,F,imageFeatures(I(:,:,band),...
                    [],[],[],[],[],[],[], [],...
                    [], [],...
                    [], [], [], model.gradient_smooth_kernel, model.tpi_kernel)); 
                fprintf('Computed features from band %d of %d in image %d of %d\n', band, nBands, imIndex, nImages);
            else 
                warning('Not using features from band %d becasue it is not included in type ''%s''', band, env.inputType)
            end
    %         F=cat(3,F,F0); clear F0;
        end
        fprintf('Classifying image %d of %d:  %s...\n',imIndex,length(imagePaths), imagePaths{imIndex});
%         try
    %         warning('off', 'MATLAB:MKDIR:DirectoryExists'); % MUTE THE
    %         WARNING using warning('on','verbose') to query warning message
    %         SOMEHOW
        [imL,classProbs] = imClassify(F,model.treeBag,nSubsets);
% %         catch e % if out of memory
%         fprintf('EK: Error during classifying:  %s\nMemory crash?\n', imagePaths{imIndex});
%         fprintf(1,'The identifier was:\t%s\n',e.identifier);
%         fprintf(1,'There was an error! The message was:\t%s\n',e.message);
%         end
        fprintf('time: %f s\n', toc(stepTime));

        [fpath,fname] = fileparts(imagePaths{imIndex});

        %% save individ masks or class probs, if selected
        for pmIndex = 1:size(classProbs,3)
            if outputMasks
                base_out=sprintf('%s_Class%02d.png',fname, pmIndex);
                fprintf('\tWriting indiv. masks:\t%s\n', base_out);
                imwrite(imL == pmIndex,[fpath filesep fname base_out]);
            end
            if outputProbMaps
                base_out=sprintf('%s_Class%02d.png',fname, pmIndex);
                fprintf('\tWriting indiv. class probs:\t%s\n', base_out);
                imwrite(classProbs(:,:,pmIndex),[fpath filesep fname base_out]);
            end
        end

        %% Write classified image
        try georef_out(fname, imL, 1);
            fprintf('Writing classified tif for:\t%s.\n', fname);
        catch
            warning('Not able to write tif:\t%s.\n', fname);
        end
    catch e
        warning('Error at some point during processing of %s.\nError ID: %s\nError message: %s',...
            imagePaths{imIndex}, e.identifier, e.message)
    end
end

fprintf('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nDone classifying.\n')
toc(totalTime)
fprintf('Output images are in: %s\n\n', env.output.test_dir)