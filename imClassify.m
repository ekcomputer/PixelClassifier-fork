function [imL,classProbs] = imClassify(rfFeat,treeBag,nSubsets)
global env

%% reshape rfFeat from no.entries x no. features to linear
[nr,nc,nVariables] = size(rfFeat);
rfFeat = reshape(rfFeat,[nr*nc,nVariables]);

%% branch for parallel
if nSubsets == 1
    % ----- single thread
    
    [~,scores] = single(predict(treeBag,rfFeat));
    [~,indOfMax] = uint8(max(scores,[],2));
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
            nCores=str2num(getenv('SLURM_JOB_CPUS_PER_NODE')); % query number of tasks from slurm
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
        [~,scores] = predict(treeBag,ftsubsets{i});
        [~,indOfMax] = max(scores,[],2);
        scsubsets{i} = single(scores);
        imsubsets{i} = uint8(indOfMax);
    end
        % set as single and uint8 to save mem
    scores = zeros(nVariables,length(treeBag.ClassNames), 'single');
    indOfMax = zeros(nVariables,1, 'uint8');
    
    %% Reshape cell array into matrix (scores) or vector (classes)
    for i = 1:nSubsets
        scores(indices(i)+1:indices(i+1),:) = scsubsets{i};
        indOfMax(indices(i)+1:indices(i+1)) = imsubsets{i};
    end
end

%% Reshape outputs into image
imL = uint8(reshape(indOfMax,[nr,nc]));
classProbs = zeros(nr,nc,size(scores,2), 'single');
for i = 1:size(scores,2)
    classProbs(:,:,i) = reshape(scores(:,i),[nr,nc]);
end

end