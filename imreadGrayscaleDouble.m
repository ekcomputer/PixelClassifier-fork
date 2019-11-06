function I = imreadGrayscaleDouble(path)
    I = imread(path);
    if size(I,3) > 1
%         for k=1:size(I,3)
%             I{k} = I(:,:,1);
%         end
%           I = rgb2gray(I); 
    end
    
    %% second loop
    if isa(I,'uint8') % TO UPDATE:
        I = double(I)/255;
        warning('did not recognize image class')
    elseif isa(I,'uint16')
        I = double(I)/65535;
        warning('did not recognize image class')
    elseif isa(I,'single')
%         warning('did not recognize image class')
    else
        warning('did not recognize image class at all')
    end
end