function Rx=Rieszfilters(Size)
%-------------------------------------------------------------
%----------- Riesz filter -----------------------------------
% A New Extension of Linear Signal Processing for
% Estimating Local Properties and Detecting Features
% Size: an even number
x = (-Size/2:Size/2)/Size; z=x';
Rx = -x.*(1./(x.^2+z.^2).^1.5); Rx(isnan(Rx))=0;Rx=Rx/max(abs(Rx(:)))*0.5;