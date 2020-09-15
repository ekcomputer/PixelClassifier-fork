function [imL,classProbs] = imClassify(rfFeat,treeBag,nSubsets)
global env

%% reshape rfFeat from no.entries x no. features to linear
[nr,nc,nVariables] = size(rfFeat);
rfFeat = reshape(rfFeat,[nr*nc,nVariables]);
nClasses=length(treeBag.ClassNames);
%% branch for parallel
if nSubsets == 1   
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
else
    % ----- parallel
    
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
        warning('Custom profile (%s) didn''t work.  Using local profile',...
            env.asc.parProfile)
    end
    
    %% classify in a parallel loop
    parfor i = 1:nSubsets
        
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
end

%% Reshape outputs into image ( could be slightly faster if vectorized, but only takes 3 sec)
imL = reshape(indOfMax,[nr,nc]); % already uint8
classProbs = zeros(nr,nc,size(scores,2), 'single');
for i = 1:size(scores,2)
    classProbs(:,:,i) = reshape(scores(:,i),[nr,nc]);
end

end