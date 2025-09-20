function y = SmoothDerivative(filterType,filterSize,signalSize)
% performs 2-d Riez transform operator (when applied to a 2-d signal
% generates a smooth derivative of it.
% y = smooth_derivatives(filterType,filterSize,signalSize)
% filterType: direction of the derivative, 'x' or 'z'
% filterSize: length of the filter, an even number
% signalSize: size of the input signal, [nz,nx]
% Ali Gholami, Institute of Geophysics, Polish Academy of Sciences, 2023
y.adjoint = 0;
[y.nff,y.nss,y.s] = setsize(filterSize+[1 1],signalSize);
R=Rieszfilters(filterSize);
if filterType=='x'
    y.R = padarray(R,y.nff,'post');
else
    y.R = padarray(R',y.nff,'post');
end
y.ns = signalSize;
y.nf = filterSize;
[y.P,y.C]=paddingmtx(y);
y = class(y,'SmoothDerivative');