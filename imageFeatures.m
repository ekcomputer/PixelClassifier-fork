function [F,featNames] = imageFeatures(I,sigmas,offsets,osSigma,radii,cfSigma,logSigmas,sfSigmas, use_raw_image, textureWindows, speckleFilter, varargin)

%% inputs
if ~isempty(varargin)
    name=varargin{1}; % names of input files (used for translation filter)
    R=varargin{2}; % map ref object
    mapinfo=varargin{3}; % map info, incl projection
    gradient_smooth_kernel=varargin{4};
    tpi_kernel=varargin{5};
    if nargin >= 17
        env = varargin{6}; % backwards compat; % avoid passing global arg
    end
else
    name='NaN'; % hidden error
    gradient_smooth_kernel=0;
    tpi_kernel=0;
end

%% create and apply data mask
msk=isnan(I); % negative mask
I(msk)=0;
F = [];
featIndex = 0;
featNames = {};
if use_raw_image
    featIndex = featIndex+1;
    featNames{featIndex} = sprintf('rawImage');
    F=cat(3, F, I); % just use original image w/o filters!
end

%% Compute features
if ~isempty(sigmas)
    derivNames = {'d0','dx','dy','dxx','dxy','dyy','hessEV1','hessEV2'};
    for sigma = sigmas
        for i = 1:length(derivNames)
            featIndex = featIndex+1;
            featNames{featIndex} = [sprintf('sigma%d',sigma) derivNames{i}];
        end
        D = zeros(size(I,1),size(I,2),8);
        [D(:,:,1),D(:,:,2),D(:,:,3),D(:,:,4),D(:,:,5),D(:,:,6),D(:,:,7),D(:,:,8)] = derivatives(I,sigma);
        F = cat(3,F,D);
        featIndex = featIndex+1;
        featNames{featIndex} = sprintf('sigma%dedges',sigma);
        F = cat(3,F,sqrt(D(:,:,2).^2+D(:,:,3).^2)); % edges
    end
end
if ~isempty(offsets)
    J = filterGauss2D(I,osSigma);
    for r = offsets
        aIndex = 0;
        try % if no name input or problem with name parsing
            heading = CalculateRangeHeading(name, R, mapinfo, env);
        catch
            heading = [pi/2, 3*pi/2]';
            warning('CalculateRangeHeading failed.  Check .ann file.  Using default of 90 and 270 deg.')
        end
        for a = heading'  % translation angles 
            aIndex = aIndex+1;
            v = r*[cos(a) sin(a)];
            T = imtranslate(J,v,'OutputView','same');
            F = cat(3,F,T);
            featIndex = featIndex+1;
            featNames{featIndex} = sprintf('offset%din%1.1f',r,heading(aIndex));
        end
    end
end

if ~isempty(radii)
    for r = radii
        [C1,C2] = circlikl(I,r,cfSigma,16,0.5);
        featIndex = featIndex+1;
        featNames{featIndex} = sprintf('circfeat%dC1',r);
        F = cat(3,F,C1);
        featIndex = featIndex+1;
        featNames{featIndex} = sprintf('circfeat%dC2',r);
        F = cat(3,F,C2);
    end
end

if ~isempty(logSigmas)
    for sigma = logSigmas
        featIndex = featIndex+1;
        featNames{featIndex} = sprintf('sigma%dlog',sigma);
        F = cat(3,F,filterLoG(I,sigma));
    end
end

if ~isempty(sfSigmas)
    for sigma = sfSigmas
        featIndex = featIndex+1;
        featNames{featIndex} = sprintf('sigma%dsteer',sigma);
        F = cat(3,F,single(steerableDetector(double(I),4,sigma))); % hot fix to allow single precision for steerable detecter
    end
end

if ~isempty(textureWindows)
   for window = textureWindows
       featIndex = featIndex+1;
       featNames{featIndex} = sprintf('window%dtexture',window);
       F = cat(3,F,movstd(I,window, 'omitnan' )); % need to mirror window...
   end
end

if ~isempty(speckleFilter)
    featIndex = featIndex+1;
    featNames{featIndex} = 'speckleFilter';
    F = cat(3,F,imguidedfilter(I));
end

if gradient_smooth_kernel>0
    global env
    addpath(env.paths.topotoolbox);
    DEM=GRIDobj([1:size(I,2)],[1:size(I,1)],I);

        % fill sinks and create flow obj
    DEMf = fillsinks(DEM); % , isnan(DEM) % note: NaN's have already been replaced with zero, so they will get filled
%     figure; imageschs(DEMf); colorbar; title('Filled DEM')

        % Smooth and Gradient
    DEMs = filter(DEMf,'mean',[gradient_smooth_kernel, gradient_smooth_kernel]);
    DEMg= gradient8(DEMs, 'per', 'useparallel', 1);
%     figure; imagesc(DEMg, [0 25]); colorbar; title('Max Gradient')
    F = cat(3,F,DEMg.Z);
    featIndex = featIndex+1;
    featNames{featIndex} = sprintf('gradSmooth%d',gradient_smooth_kernel);
end

if tpi_kernel > 0
        %% TPI
    DEMr= roughness(DEMf, 'tpi', [tpi_kernel, tpi_kernel]);
%     figure; imagesc(DEMr, [-0.4 0.4]); colorbar; title('TPI')
    F = cat(3,F,DEMr.Z);
    featIndex = featIndex+1;
    featNames{featIndex} = sprintf('TPI%dkernel',tpi_kernel);
end
%% add back in NaNs
F(repmat(msk,[1,1,size(F,3)]))=NaN;

end