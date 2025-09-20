function [f,g] = smoothsmoothulwithgrad_pa(gamma, G, Y, epsilon, xi, Dx, Dz,Dxs, Dzs, nx,nz, beta, sigma1, sigma2)


nxnz = nx*nz;
D = [sigma1*(cos(gamma(1:nxnz)).*Dx + sin(gamma(1:nxnz)).*Dz)
     sigma2*(-sin(gamma(1:nxnz)).*Dx + cos(gamma(1:nxnz)).*Dz)];

Ds =@(m) [sigma1*(cos(gamma(1:nxnz)).*(Dxs*m) + sin(gamma(1:nxnz)).*(Dzs*m))
                 sigma2*(-sin(gamma(1:nxnz)).*(Dxs*m) + cos(gamma(1:nxnz)).*(Dzs*m))];

Ds_adj =@(dm) [sigma1*(Dxs'*(cos(gamma(1:nxnz)).*dm(1:nxnz)) + Dzs'*(sin(gamma(1:nxnz)).*dm(1:nxnz)))+...
                 sigma2*(Dxs'*(-sin(gamma(1:nxnz)).*dm(nxnz+(1:nxnz))) + Dzs'*(cos(gamma(1:nxnz)).*dm(nxnz+(1:nxnz))))];


coeffmatr = G'*G + gamma(nxnz+1)*(D'*D) + 1e-10*speye(size(G,2));
if issparse(coeffmatr)
    [L,U,P,Q] = lu(coeffmatr);
    coeffmatr =@(x) Q*(U\(L\(P*x)));
else
    [L,U,P] = lu(coeffmatr);
    coeffmatr =@(x) U\(L\(P*x));
end
    
        
Nabla = [Dx;Dz];

m = coeffmatr(G'*Y(:));

f = 0.5*xi*(norm(G*m - Y(:))^2 - epsilon)^2 ...
                   + 0.5*norm(Ds(m))^2 + 0.5*beta*norm(Nabla*(gamma(1:nxnz)))^2; 

         
               
if nargout > 1
    %                 R = [cos, sin  ]              R' = [cos, -sin  ]
    %                        [-sin, cos]                       [sin, cos]
    %                 dR = [-sin, cos ]           dR' =[-sin, -cos]
    %                         [-cos, -sin]                    [cos, -sin] 
    mx = (Dxs*m); mz = Dzs*m;
    a(:,1) = sigma1^2*(cos(gamma(1:nxnz)).*mx +  sin(gamma(1:nxnz)).*mz);
    a(:,2) = sigma2^2*(-sin(gamma(1:nxnz)).*mx  +  cos(gamma(1:nxnz)).*mz);
    b(:,1) =  -sin(gamma(1:nxnz)).*a(:,1) - cos(gamma(1:nxnz)).*a(:,2);
    b(:,2) =  cos(gamma(1:nxnz)).*a(:,1) - sin(gamma(1:nxnz)).*a(:,2);
    grad1_1 = mx.*b(:,1)+ mz.*b(:,2);
    a(:,1) = sigma1^2*(-sin(gamma(1:nxnz)).*mx +  cos(gamma(1:nxnz)).*mz);
    a(:,2) = sigma2^2*(-cos(gamma(1:nxnz)).*mx  -  sin(gamma(1:nxnz)).*mz);
    b(:,1) =  cos(gamma(1:nxnz)).*a(:,1) - sin(gamma(1:nxnz)).*a(:,2);
    b(:,2) =  sin(gamma(1:nxnz)).*a(:,1) + cos(gamma(1:nxnz)).*a(:,2);
    grad1_1 = grad1_1 +  mx.*b(:,1) + mz.*b(:,2); 
    grad1_1 = 0.5*grad1_1;
    grad1_1(nxnz+1)=0;
    
    
    grad1_2 = beta*[Nabla'*Nabla*gamma(1:nxnz); 0];
    grad1 = grad1_1 + grad1_2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    grad2_1 = 2*xi*(norm(G*m - Y(:))^2 - epsilon)*(G'*(G*m - Y(:)));
    grad2_2 = Ds_adj(Ds(m));
    grad2_1p2_2 = coeffmatr(grad2_1 + grad2_2);
    
    mx = (Dx*m); mz = Dz*m;
    a(:,1) = sigma1^2*(cos(gamma(1:nxnz)).*mx +  sin(gamma(1:nxnz)).*mz);
    a(:,2) = sigma2^2*(-sin(gamma(1:nxnz)).*mx  +  cos(gamma(1:nxnz)).*mz);
    b(:,1) = -sin(gamma(1:nxnz)).*a(:,1) - cos(gamma(1:nxnz)).*a(:,2);
    b(:,2) = cos(gamma(1:nxnz)).*a(:,1) - sin(gamma(1:nxnz)).*a(:,2);
    Jac = (Dx*grad2_1p2_2).*b(:,1) + (Dz*grad2_1p2_2).*b(:,2);
    a(:,1) = sigma1^2*(-sin(gamma(1:nxnz)).*mx +  cos(gamma(1:nxnz)).*mz);
    a(:,2) = sigma2^2*(-cos(gamma(1:nxnz)).*mx  -  sin(gamma(1:nxnz)).*mz);
    b(:,1) =  cos(gamma(1:nxnz)).*a(:,1) - sin(gamma(1:nxnz)).*a(:,2);
    b(:,2) =  sin(gamma(1:nxnz)).*a(:,1) + cos(gamma(1:nxnz)).*a(:,2);
    Jac = Jac +   (Dx*grad2_1p2_2).*b(:,1) + (Dz*grad2_1p2_2).*b(:,2); 
    
    grad2 = [-gamma(nxnz+1)*(Jac);  -grad2_1p2_2'*((D'*D)*m)];
    
    g = grad1 + grad2;
    
end
