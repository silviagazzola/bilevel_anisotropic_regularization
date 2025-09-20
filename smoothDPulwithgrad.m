function [f,g] = smoothDPulwithgrad(gamma, G, Y, epsilon, Dx, Dz)


D = [Dx; Dz];

coeffmatr = G'*G + gamma*(D'*D);

m = coeffmatr\(G'*Y(:));

f = 0.5*(norm(G*m - Y(:))^2 - epsilon)^2 ;

if nargout > 1 % gradient required
    Jac = - coeffmatr\((D'*D)*m);
    
    g = 2*(norm(G*m - Y(:))^2 - epsilon)*(G'*(G*m - Y(:)));
    g = Jac'*g;
end


