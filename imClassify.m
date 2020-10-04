function [imL,classProbs] = imClassify(rfFeat,treeBag,nSubsets)
% Re-written to take a .tif file pathname instead of matrix as 'rfFeat', if memory
% constraints. Old option (matrix as input) is still retained.
% TODO: modify original code for parallel to not load entire mat-obj of F
global env
block_proc=0;
if isstr(rfFeat)
    block_proc = exist(rfFeat)==2; % if input 'rfFeat' is a pathname (use block processing to file)
end
%% reshape rfFeat from no.entries x no. features to linear

%% branch for parallel
if nSubsets == 1   
        % Reshape
    [nr,nc,nVariables] = size(rfFeat);
    rfFeat = reshape(rfFeat,[nr*nc,nVariables]);
    nClasses=length(treeBag.ClassNames);

    % pre-allocate answer
    scores=zeros([size(rfFeat, 1), length(treeBag.ClassNames)], 'like', rfFeat);
    indOfMax=uint8(env.constants.noDataValue_ouput)*ones([size(rfFeat, 1),1], 'uint8'); % same as zeros

        % generate mask
    msk=any(isnan(rfFeat),2); % negative mask

        % predict only on valid data
    [~,scores_valid] = predict(treeBag,rfFeat(~msk,:));
    [~,indOfMax_valid] = max(scores_valid,[],2);

        % add in classified valid data to pre-allocated zeros, using mask

    scores(~msk,:)=scores_valid;  
    indOfMax(~msk)=indOfMax_valid;
elseif nSubsets > 1 & ~block_proc
    % ----- parallel
    
        % Reshape
    [nr,nc,nVariables] = size(rfFeat);
    rfFeat = reshape(rfFeat,[nr*nc,nVariables]);
    nClasses=length(treeBag.ClassNames);
    
    indices = round(linspace(0,size(rfFeat,1),nSubsets+1));

    ftsubsets = cell(1,nSubsets);
    for i = 1:nSubsets
        ftsubsets{i} = rfFeat(indices(i)+1:indices(i+1),:);
    end
    clear rfFeat % save memory
    
    scsubsets = cell(1,nSubsets);
    imsubsets = cell(1,nSubsets);
    
    %% start parallel pool
    try
        if isunix && isempty(gcp('nocreate')) % if on ASC and no pool running
            nCores=str2double(getenv('SLURM_JOB_CPUS_PER_NODE')); % query number of tasks from slurm
            if isempty(nCores) % not in slurm environment
                nCores=8;
            end
            parpool(env.asc.parProfile, nCores)
        else
%             parpool % use default
        end
        
    catch
        fprintf('EK warning: Custom profile (%s) didn''t work.  Using local profile',...
            env.asc.parProfile)
    end
    
    %% classify in a parallel loop
    for i = 1:nSubsets % TODO: change back to parfor
        
            % pre-allocate answer
        scores=zeros([size(ftsubsets{i}, 1), nClasses], 'like', ftsubsets{i});
        indOfMax=uint8(env.constants.noDataValue_ouput)*ones([size(ftsubsets{i}, 1),1], 'uint8'); % same as zeros
        
            % generate mask
        msk=any(isnan(ftsubsets{i}),2); % negative mask
        
            % predict only on valid data
        [~,scores_valid] = predict(treeBag,ftsubsets{i}(~msk,:));
        [~,indOfMax_valid] = max(scores_valid,[],2);
        
            % add in classified valid data to pre-allocated zeros, using mask
            
        scores(~msk,:)=scores_valid;  
        indOfMax(~msk)=indOfMax_valid;
            
            % save mem
        scsubsets{i} = scores; % already single
        imsubsets{i} = indOfMax; % already uint8
    end
    
    %% pre-allocate and re-use variables; set as single and uint8 to save mem    
    scores = zeros(nVariables,length(treeBag.ClassNames), 'single');
    indOfMax = zeros(nVariables,1, 'uint8'); % testing
    
    %% Reshape cell array into matrix (scores) or vector (classes)
    for i = 1:nSubsets
        scores(indices(i)+1:indices(i+1),:) = scsubsets{i};
        indOfMax(indices(i)+1:indices(i+1)) = imsubsets{i};
    end
elseif nSubsets > 1 & block_proc
    info = imfinfo(rfFeat); % get size
%     [f.y,f.x,f.z]=size(info.F); % can be slow
    f.y=info.Height;
    f.x=info.Width;
%     f.el=f.x*f.y*f.z;
    f.tile_size= 1024; %round(sqrt(f.el/nSubsets)); % HERE: optimize
%     out_pth=[env.tempDir, 'cls_tmp.tif'];
    outFileWriter = BP_bigTiffWriterEK([rfFeat(1:end-10), '_cls.tif'], f.y, f.x, f.tile_size, f.tile_size); % remove the '_bands.tif'
    func = @(block_struct)  imClassify_function(block_struct, treeBag, env.constants.noDataValue_ouput);
    % HERE: need to convert matfile to tif for an image that's too big to
    % fit in memory...
    blockproc(rfFeat, [f.tile_size, f.tile_size], func, 'destination', outFileWriter, 'UseParallel', true);
    outFileWriter.close();
    % NOTE georef info is written in parent function
    indOfMax='already-written'; % HERE
    return
else
    error('Error EK.')
end

if ~block_proc
    %% Reshape outputs into image ( could be slightly faster if vectorized, but only takes 3 sec)
    imL = reshape(indOfMax,[nr,nc]); % already uint8
    classProbs = zeros(nr,nc,size(scores,2), 'single');
    for i = 1:size(scores,2)
        classProbs(:,:,i) = reshape(scores(:,i),[nr,nc]);
    end
end

end