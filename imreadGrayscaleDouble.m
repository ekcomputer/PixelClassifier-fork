function I = imreadGrayscaleDouble(path)
    I = imread(path);
    if size(I,3) > 1
%         for k=1:size(I,3)
%             I{k} = I(:,:,1);
%         end
    end
    if isa(I,'uint8') % TO UPDATE:
        I = double(I)/255;
    elseif isa(I,'uint16')
        I = double(I)/65535;
    elseif isa(I,'single')
    else
        warning('did not recognize image class')
    end
end