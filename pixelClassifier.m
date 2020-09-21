% Note "if 1==1 switch to avoid using blockproc"
clear, 
% clc
fprintf('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nStarting classification queue...\n')
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
        try
            [I,R] = geotiffread(imagePaths{imIndex});
            mapinfo=geotiffinfo(imagePaths{imIndex});
            if isempty(R) || isempty(mapinfo.SpatialRef)
                error('EK: Empty R')
            end
        catch w
            warning('EK Warning: initial geotifinfo failed. Using gdaledit and retrying...\n\tFile: %s', imagePaths{imIndex})
                
                % use gdal_edit and values from other matlab I/O functions
                % to re-assign SRS info if image is normal tif + tfw and
                % proj
            worldfile=[imagePaths{imIndex}(1:end-4), '.tfw'];
            im_info=imfinfo(imagePaths{imIndex});
            co=worldfileread(worldfile, 'planar', [im_info.Height, im_info.Width]);
            bb=[co.XWorldLimits(1), co.YWorldLimits(2), co.XWorldLimits(2), co.YWorldLimits(1)];
            cmd=sprintf('gdal_edit.py -a_srs "EPSG:102001" -a_ullr %s %s', num2str(bb), imagePaths{imIndex});
            system(cmd);
            [I,R] = geotiffread(imagePaths{imIndex});
        end
        mapinfo=geotiffinfo(imagePaths{imIndex});
        nBands=size(I, 3);

            % set NoData values to NaN
        I(repmat(any(I(:, :, env.radar_bands)<0,3), [1,1, size(I,3)]))=NaN; % set pixels to NaN if any radar band <0   
        I(repmat(all(I(:, :,env.radar_bands)==env.constants.noDataValue, 3), [1,1, size(I,3)]))=NaN; % set pixels to NaN if each radar band ==0

            % remove NaN's
    % % Note section under heading 'mask out near range, if applicable' adds
    % NaNs back in...
    %     I(repmat(isnan(I(:,:,nBands)),...
    %         [1, 1, nBands]))=env.constants.noDataValue;
        tic;

        %% mask out near range, if applicable
        if ~isnan(env.inc_band) & env.IncMaskMin> 0 % if input type doesn't use inc as feature, mask out near range inc angles bc they are unreliable
            if size(I, 3) <4 % no inc band was included
                error('No inc. band found?')
            else % inc band was included
                fprintf('Masking out inc. angle < %0.2f.\n', env.IncMaskMin)
                msk=I(:,:,env.inc_band) < env.IncMaskMin; % negative mask for near range
                I(repmat(msk, [1,1, nBands]))=NaN;  % BW=logical(repmat(BW, [1 1 3]));
            end
        end
        %% loop over bands w/i image

        F=single.empty(size(I,1),size(I,2),0); % initilize
        F_obj_pth=[env.tempDir, 'imageFeatures_bands.mat'];
        delete(F_obj_pth); % in case it exists from prev image
        save(F_obj_pth, 'F', '-v7.3') % initialize matfile
        F_object = matfile(F_obj_pth,'Writable',true); % initialize matfile object
        for band=1:nBands
            if ismember(band, env.radar_bands) % for radar bands
                F = imageFeatures(I(:,:,band),model.sigmas,...
                    model.offsets,model.osSigma,model.radii,model.cfSigma,...
                    model.logSigmas,model.sfSigmas, model.use_raw_image,...
                    model.textureWindows, model.speckleFilter,...
                    names{imIndex}, R, mapinfo,...
                    [],[]);
                fprintf('Computed features from band %d of %d in image %d of %d', band, nBands, imIndex, nImages);
            elseif ismember(band, env.inc_band) % for incidence angle band
                F = imageFeatures(I(:,:,band),[],[],[],[],[],[],[], 1, [], [],...
                    [],[],[],[],[]);
                fprintf('Computed features from band %d of %d in image %d of %d', band, nBands, imIndex, nImages);
    %         elseif band == nBands && ismember(env.inputType, {'Freeman', 'C3', 'T3', 'Sinclair'})
                % Don't extract any features from inc. band.   
            elseif ismember(band, env.dem_band) % for DEM/hgt band
                F = imageFeatures(I(:,:,band),...
                    [],[],[],[],[],[],[], [],...
                    [], [],...
                    [], [], [], model.gradient_smooth_kernel, model.tpi_kernel); 
                fprintf('Computed features from band %d of %d in image %d of %d', band, nBands, imIndex, nImages);
            else 
                warning('Not using features from band %d becasue it is not included in type ''%s''', band, env.inputType)
                continue
            end
            [f.y,f.x,f.z]=size(F_object.F);
            F_object.F(:,:,f.z+1:f.z+size(F,3))=F; % write out to F_object
            fprintf('  ---> saved\n')
            clear F
        end % end loop over bands
        clear('F_object')
        fprintf('Classifying image %d of %d:  %s...\n',imIndex,length(imagePaths), imagePaths{imIndex});
        try
                % load entire F file into memory - hopefully, it fits...
                % should be 30 GB for a 22x23 kpx image with 15 feature
                % bands...
            F=load(F_obj_pth);
            [imL,classProbs] = imClassify(F,model.treeBag,nSubsets);
        catch e % if out of memory
            fprintf('EK: Error during classifying:  %s\nMemory crash?\n', imagePaths{imIndex});
            fprintf(1,'The identifier was:\t%s\n',e.identifier);
            fprintf(1,'There was an error! The message was:\t%s\n',e.message);
        end
        fprintf('time: %f s\n', toc);

        [fpath,fname] = fileparts(imagePaths{imIndex});
        
        %% Proceed to write output

        if 1 == 1
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

                % Write classified image
            try georef_out(fname, imL, true);
                fprintf('Writing classified tif for:\t%s.\n', fname);
            catch
                warning('Not able to write tif:\t%s.\n', fname);
            end
        else % Depricated: write in chuncks not working with .mat file
                % movefiles
            [DESTINATION, gt]=georef_out(fname, imL, false);
            [SUCCESS,MESSAGE,MESSAGEID] = movefile([env.tempDir, 'cls_tmp.tif'],DESTINATION)
            if SUCCESS==0
               error(MESSAGE) 
            end
            
                % write georef info
            gti_out=[DESTINATION(1:end-4), '.tfw']; % Move up one in stack
            worldfilewrite(gt.SpatialRef, gti_out);
            % TODO: write proj file HERE?
        end
    catch e
        warning('Error at some point during processing of %s.\nError ID: %s\nError message: %s\nStack:\n',...
            imagePaths{imIndex}, e.identifier, e.message)
        disp(e.stack)
    end
end

fprintf('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nDone classifying.\n')
toc
fprintf('Output images are in: %s\n\n', env.output.test_dir)