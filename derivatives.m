function [d0,dx,dy,dxx,dxy,dyy,eigVal1,eigVal2] = derivatives(I,sigma)

s = sigma;

w = ceil(4*s);
x = -w:w;

g = exp(-x.^2/(2*s^2)) / (sqrt(2*pi)*s); % gaussian
gx = -x/s^2 .* g; % first deriv
gxx = x.^2 .* g / s^4; % -1/s^2 term subtracted below % second deriv

inputXT = padarray(I, [w w], 'symmetric');

d0 = conv2(g, g, inputXT, 'valid') / s^2; % col, row kernel

dxx = conv2(g, gxx, inputXT, 'valid') - d0;
dxy = conv2(gx, gx, inputXT, 'valid');
dyy = conv2(gxx, g, inputXT, 'valid') - d0;

dx = conv2(g,gx,inputXT,'valid');
dy = conv2(gx,g,inputXT,'valid');

alpha = (dxx + dyy)/2;
beta = sqrt((dxx - dyy) .^2 + 4*dxy .^2)/2;
eigVal1 = alpha+beta;
eigVal2 = alpha-beta;

end