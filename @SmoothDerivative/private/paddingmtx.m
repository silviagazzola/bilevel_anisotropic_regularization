function [P,C]=paddingmtx(y)

nxnz = prod(y.ns);
idx = reshape(1:nxnz,y.ns);
idx = padarray(idx,y.nss(1,:),'pre','symmetric');
idx = padarray(idx,y.nss(2,:),'post','symmetric');
N = length(idx(:));
P = sparse(1:N,idx(:),ones(N,1),N,nxnz);

idx = zeros(y.ns+sum(y.nss)); idx((1:y.ns(1))+y.s(1),(1:y.ns(2))+y.s(2),:)=1;
C = speye(length(idx(:)));
C(idx==0,:)=[];