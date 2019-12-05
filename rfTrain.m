function [treeBag,featImp,oobPredError] = rfTrain(rfFeat,rfLbl,ntrees,minleafsize, varargin)
% varargin gives optional random number generator seed
% ntrees = 20; minleafsize = 60;
opt = statset('UseParallel',false); % set true to use parallel for large datasets
if isnumeric(varargin{1}) || strcmp(varargin{1}, 'shuffle')
    rng(varargin{1}); % set seed for random # generator to replicate training each time
end
treeBag = TreeBagger(ntrees,rfFeat,rfLbl,'MinLeafSize',minleafsize,'oobvarimp','on','opt',opt);
if nargout > 1
    featImp = treeBag.OOBPermutedVarDeltaError;
end
if nargout > 2
    oobPredError = oobError(treeBag);
end

end