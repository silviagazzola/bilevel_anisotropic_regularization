function [f,g] = smoothulwithgrad(gamma, G, Y, epsilon, xi, Dx, Dz, Dxs, Dzs, nx, nz, beta1, beta2)

nxnz = nx*nz;

sigma1 = gamma(nxnz+1:2*nxnz);
sigma2 = 1 - sigma1;

D = [sigma1.*(cos(gamma(1:nxnz)).*Dx + sin(gamma(1:nxnz)).*Dz)
     sigma2.*(-sin(gamma(1:nxnz)).*Dx + cos(gamma(1:nxnz)).*Dz)];

Ds =@(m) [sigma1.*(cos(gamma(1:nxnz)).*(Dxs*m) + sin(gamma(1:nxnz)).*(Dzs*m))
          sigma2.*(-sin(gamma(1:nxnz)).*(Dxs*m) + cos(gamma(1:nxnz)).*(Dzs*m))];

Ds_adj =@(dm) Dxs'*(cos(gamma(1:nxnz)).*(sigma1.*dm(1:nxnz))) +... 
              Dzs'*(sin(gamma(1:nxnz)).*(sigma1.*dm(1:nxnz))) +...
              Dxs'*(-sin(gamma(1:nxnz)).*(sigma2.*dm(nxnz+(1:nxnz)))) +... 
              Dzs'*(cos(gamma(1:nxnz)).*(sigma2.*dm(nxnz+(1:nxnz))));


coeffmatr = G'*G + gamma(2*nxnz+1)*(D'*D);
[L,U,P] = lu(coeffmatr);
coeffmatr =@(x) (U\(L\(P*x)));
        
Nabla = [Dx;Dz];

m = coeffmatr(G'*Y(:));

f = 0.5*xi*(norm(G*m - Y(:))^2 - epsilon)^2 ...
                   + 0.5*norm(Ds(m))^2 ... 
                   + 0.5*beta1*norm(Nabla*(gamma(1:nxnz)))^2 ...
                   + 0.5*beta2*norm(Nabla*(gamma(nxnz+1:2*nxnz)))^2;

         
               
if nargout > 1
    %                 R = [cos, sin  ]              R' = [cos, -sin  ]
    %                        [-sin, cos]                       [sin, cos]
    %                 dR = [-sin, cos ]           dR' =[-sin, -cos]
    %                         [-cos, -sin]                    [cos, -sin] 

    mx = (Dxs*m); mz = Dzs*m;
    a(:,1) = sigma1.^2.*(cos(gamma(1:nxnz)).*mx +  sin(gamma(1:nxnz)).*mz);
    a(:,2) = sigma2.^2.*(-sin(gamma(1:nxnz)).*mx  +  cos(gamma(1:nxnz)).*mz);
    b(:,1) =  -sin(gamma(1:nxnz)).*a(:,1) - cos(gamma(1:nxnz)).*a(:,2);
    b(:,2) =  cos(gamma(1:nxnz)).*a(:,1) - sin(gamma(1:nxnz)).*a(:,2);
    grad1_1 = mx.*b(:,1)+ mz.*b(:,2);
    a(:,1) = sigma1.^2.*(-sin(gamma(1:nxnz)).*mx +  cos(gamma(1:nxnz)).*mz);
    a(:,2) = sigma2.^2.*(-cos(gamma(1:nxnz)).*mx  -  sin(gamma(1:nxnz)).*mz);
    b(:,1) =  cos(gamma(1:nxnz)).*a(:,1) - sin(gamma(1:nxnz)).*a(:,2);
    b(:,2) =  sin(gamma(1:nxnz)).*a(:,1) + cos(gamma(1:nxnz)).*a(:,2);
    grad1_1 = grad1_1 +  mx.*b(:,1) + mz.*b(:,2); 
    grad1_1 = 0.5*grad1_1; %%% WHY WAS THIS INCLUDED??? (probably because we are deriving wrt a parameter in a matrix)
    
    grad1_2 = beta1*Nabla'*(Nabla*gamma(1:nxnz));
    grad1 = grad1_1 + grad1_2;

    % % new bits
    % a(:,1) = sigma1.*(cos(gamma(1:nxnz)).*mx +  sin(gamma(1:nxnz)).*mz);
    % a(:,2) = sigma2.*(-sin(gamma(1:nxnz)).*mx  +  cos(gamma(1:nxnz)).*mz);
    % b(:,1) =  cos(gamma(1:nxnz)).*a(:,1) - sin(gamma(1:nxnz)).*a(:,2);
    % b(:,2) =  -sin(gamma(1:nxnz)).*(a(:,1)) - cos(gamma(1:nxnz)).*(a(:,2));
    % grad1s_1 = mx.*b(:,1)+ mz.*b(:,2);
    % a(:,1) = sigma1.*(cos(gamma(1:nxnz)).*mx +  sin(gamma(1:nxnz)).*mz);
    % a(:,2) = -sigma2.*(-sin(gamma(1:nxnz)).*mx  +  cos(gamma(1:nxnz)).*mz);
    % b(:,1) =  cos(gamma(1:nxnz)).*a(:,1) - sin(gamma(1:nxnz)).*a(:,2);
    % b(:,2) =  sin(gamma(1:nxnz)).*a(:,1) + cos(gamma(1:nxnz)).*a(:,2);
    % grad1s_1 = grad1s_1 +  mx.*b(:,1) + mz.*b(:,2); 
    % grad1s_1 = 0.5*grad1s_1;

    % new bits
    a(:,1) = sigma1.*(cos(gamma(1:nxnz)).*mx +  sin(gamma(1:nxnz)).*mz);
    a(:,2) = -sigma2.*(-sin(gamma(1:nxnz)).*mx  +  cos(gamma(1:nxnz)).*mz);
    b(:,1) =  cos(gamma(1:nxnz)).*a(:,1) - sin(gamma(1:nxnz)).*a(:,2);
    b(:,2) =  sin(gamma(1:nxnz)).*(a(:,1)) + cos(gamma(1:nxnz)).*(a(:,2));
    grad1s_1 = mx.*b(:,1)+ mz.*b(:,2);
    % a(:,1) = sigma1.*(cos(gamma(1:nxnz)).*mx +  sin(gamma(1:nxnz)).*mz);
    % a(:,2) = -sigma2.*(-sin(gamma(1:nxnz)).*mx  +  cos(gamma(1:nxnz)).*mz);
    % b(:,1) =  cos(gamma(1:nxnz)).*a(:,1) - sin(gamma(1:nxnz)).*a(:,2);
    % b(:,2) =  sin(gamma(1:nxnz)).*a(:,1) + cos(gamma(1:nxnz)).*a(:,2);
    % grad1s_1 = grad1s_1 +  mx.*b(:,1) + mz.*b(:,2); 
    % grad1s_1 = 0.5*grad1s_1;

    grad1s_2 = beta2*Nabla'*(Nabla*gamma(nxnz+1:2*nxnz));
    grad1s = grad1s_1 + grad1s_2;
    
    % full first term in the gradient
    grad1 = [grad1; grad1s; 0];
    % grad1 = [0.5*grad1; 0.5*grad1s; 0];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    grad2_1 = 2*xi*(norm(G*m - Y(:))^2 - epsilon)*(G'*(G*m - Y(:)));
    grad2_2 = Ds_adj(Ds(m));
    grad2_1p2_2 = coeffmatr(grad2_1 + grad2_2);
    
    mx = (Dx*m); mz = Dz*m;

    a(:,1) = sigma1.^2.*(cos(gamma(1:nxnz)).*mx +  sin(gamma(1:nxnz)).*mz);
    a(:,2) = sigma2.^2.*(-sin(gamma(1:nxnz)).*mx  +  cos(gamma(1:nxnz)).*mz);
    b(:,1) = -sin(gamma(1:nxnz)).*a(:,1) - cos(gamma(1:nxnz)).*a(:,2);
    b(:,2) = cos(gamma(1:nxnz)).*a(:,1) - sin(gamma(1:nxnz)).*a(:,2);
    Jac1 = (Dx*grad2_1p2_2).*b(:,1) + (Dz*grad2_1p2_2).*b(:,2);
    a(:,1) = sigma1.^2.*(-sin(gamma(1:nxnz)).*mx +  cos(gamma(1:nxnz)).*mz);
    a(:,2) = sigma2.^2.*(-cos(gamma(1:nxnz)).*mx  -  sin(gamma(1:nxnz)).*mz);
    b(:,1) =  cos(gamma(1:nxnz)).*a(:,1) - sin(gamma(1:nxnz)).*a(:,2);
    b(:,2) =  sin(gamma(1:nxnz)).*a(:,1) + cos(gamma(1:nxnz)).*a(:,2);
    Jac1 = Jac1 +   (Dx*grad2_1p2_2).*b(:,1) + (Dz*grad2_1p2_2).*b(:,2); 

    % % new bits
    % a(:,1) = sigma1.*(cos(gamma(1:nxnz)).*mx +  sin(gamma(1:nxnz)).*mz);
    % a(:,2) = sigma2.*(-sin(gamma(1:nxnz)).*mx  +  cos(gamma(1:nxnz)).*mz);
    % b(:,1) =  cos(gamma(1:nxnz)).*a(:,1) - sin(gamma(1:nxnz)).*a(:,2);
    % b(:,2) =  -sin(gamma(1:nxnz)).*(a(:,1)) - cos(gamma(1:nxnz)).*(a(:,2));
    % Jac2 = (Dx*grad2_1p2_2).*b(:,1) + (Dz*grad2_1p2_2).*b(:,2); 
    % a(:,1) = sigma1.*(cos(gamma(1:nxnz)).*mx +  sin(gamma(1:nxnz)).*mz);
    % a(:,2) = -sigma2.*(-sin(gamma(1:nxnz)).*mx  +  cos(gamma(1:nxnz)).*mz);
    % b(:,1) =  cos(gamma(1:nxnz)).*a(:,1) - sin(gamma(1:nxnz)).*a(:,2);
    % b(:,2) =  sin(gamma(1:nxnz)).*a(:,1) + cos(gamma(1:nxnz)).*a(:,2);
    % Jac2 = Jac2 +   (Dx*grad2_1p2_2).*b(:,1) + (Dz*grad2_1p2_2).*b(:,2); 

    % new bits
    a(:,1) = sigma1.*(cos(gamma(1:nxnz)).*mx +  sin(gamma(1:nxnz)).*mz);
    a(:,2) = -sigma2.*(-sin(gamma(1:nxnz)).*mx  +  cos(gamma(1:nxnz)).*mz);
    b(:,1) =  cos(gamma(1:nxnz)).*a(:,1) - sin(gamma(1:nxnz)).*a(:,2);
    b(:,2) =  sin(gamma(1:nxnz)).*(a(:,1)) + cos(gamma(1:nxnz)).*(a(:,2));
    Jac2 = (Dx*grad2_1p2_2).*b(:,1) + (Dz*grad2_1p2_2).*b(:,2); 

    a(:,1) = sigma1.*(cos(gamma(1:nxnz)).*mx +  sin(gamma(1:nxnz)).*mz);
    a(:,2) = -sigma2.*(-sin(gamma(1:nxnz)).*mx  +  cos(gamma(1:nxnz)).*mz);
    b(:,1) =  cos(gamma(1:nxnz)).*a(:,1) - sin(gamma(1:nxnz)).*a(:,2);
    b(:,2) =  sin(gamma(1:nxnz)).*a(:,1) + cos(gamma(1:nxnz)).*a(:,2);
    Jac2 = Jac2 +   (Dx*grad2_1p2_2).*b(:,1) + (Dz*grad2_1p2_2).*b(:,2); 

    grad2 = [-gamma(2*nxnz+1)*(Jac1); -gamma(2*nxnz+1)*(Jac2); -grad2_1p2_2'*((D'*D)*m)];
    
    g = grad1 + grad2;
    
end
