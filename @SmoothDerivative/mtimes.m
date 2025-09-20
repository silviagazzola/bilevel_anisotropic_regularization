function X = mtimes(A,x)

% for 2D Riez transform 

x = x(:);
%

if isa(A,'SmoothDerivative') == 0
    error('In  A.*B only A can be derivative operator');
end

if A.adjoint
    X = reshape(A.C'*x,A.ns+sum(A.nss));
    X=real(ifft2(bsxfun(@times,fft2(X),conj(fft2(A.R)))));
    X = A.P'*X(:);
else
    X = reshape(A.P*x,A.ns+sum(A.nss));
    X = real(ifft2(bsxfun(@times,fft2(X),fft2(A.R))));
    X = A.C*X(:);
end
