% Denoising test problem via nonstationary anisotropic Tikhonov regularization 
% Reference paper: 
% S.Gazzola and A.Gholami, Automatic nonstationary anisotropic Tikhonov regularization through bilevel optimization, 2025
% 
% S.Gazzola and A.Gholami, August 2025

% Make sure the L-BFGS-B codes from https://github.com/stephenbeckr/L-BFGS-B-C are in the MATLAB path; 
% in order to compute the whole quantities (relative errors, discrepancies, magnitude of directional derivatives ets.) at each iteration,
% take lbfgsb.m and replace the command in l.212 by:
% history(outerIter,j+2) = errFcn{j}(x);

clear, clc, close all

% load test data
load('Clapp_model_resized')

% add white Gaussian noise
sigma = 0.75;
Y = X(:) + randn(size(X(:)))*sigma;
epsilon = norm(Y-X(:),'fro')^2; % noise magnitude (delta in the paper)
nl = norm(Y(:) - X(:))/norm(X(:));

[nz,nx]=size(X);
nxnz = nz*nx;

G = speye(nxnz);

% gradient discretization
D1= sparse(toeplitz([1 zeros(1,nz-1)],[1 -1 zeros(1,nz-2)])); D1(end,end) = 0;
Dz  = kron(speye(nx),D1);
D1= sparse(toeplitz([1 zeros(1,nx-1)],[1 -1 zeros(1,nx-2)])); D1(end,end) = 0;
Dx = kron(D1,speye(nz));
D = [Dx; Dz];

% Hilbert transform (for gradients in the upper level functional)
Dxs = SmoothDerivative('x',6,[nz,nx]);
Dzs = SmoothDerivative('z',6,[nz,nx]);

% weights for the upper-level objective function
xi = 0.1; % corresponds to 1/alpha in the paper
beta1 = 0.5; % corresponds to beta1*xi in the paper
beta2 = beta1; % corresponds to beta2*xi in the paper
sigma1=0.9; sigma2 = 1e-1; % anisotropy weights for reduced upper level functional

%% compare with Tikhonov with std derivative and reg parameter chosen according to the discrepancy principle
fcn_DP = @(gamma)smoothDPulwithgrad(gamma, G, Y, epsilon, Dx, Dz);
coeffmatr_DP = @(x)(G'*G + x*(D'*D));
mx_DP = @(x)coeffmatr_DP(x)\(G'*Y(:));

% bounds for upper level variable
l  = 0; 
u  = Inf;    

% options for L-BFGS-B (upper level solver)
opts = struct('x0', 1e-4, 'factr', 1e4, 'pgtol', 1e-8, 'm', 10, 'maxIts', 100);

opts.printEvery     = 1;
eFcn = cell(2,1);
eFcn{1} = @(x) x(end);
eFcn{2} = @(x) norm(mx_DP(x)-X(:))/norm(X(:));
eFcn{3} = @(x) norm(mx_DP(x) - Y(:));
opts.errFcn = eFcn;

% Run the algorithm:
[xk_DP, fk_DP, info_DP] = lbfgsb(fcn_DP, l, u, opts);

gamma_DP = xk_DP;
coeffmatr_DP = G'*G + gamma_DP*(D'*D);
m_DP = coeffmatr_DP\(G'*Y(:)); % reconstruction

%% compare with `partially' anisotropic (pa) Tikhonov with std derivative and reg parameter chosen according to the discrepancy principle
fcn = @(gamma)smoothulwithgrad_pa(gamma, G, Y(:), epsilon, xi, Dx, Dz,Dxs, Dzs, nx,nz, beta2, sigma1, sigma2);
D = @(x)[sigma1*(cos(x(1:nxnz)).*Dx + sin(x(1:nxnz)).*Dz)
         sigma2*(-sin(x(1:nxnz)).*Dx + cos(x(1:nxnz)).*Dz)];
coeffmatr = @(x)(G'*G + x(end)*((D(x))'*(D(x))));
mx = @(x)coeffmatr(x)\(G'*Y(:));
Ds = @(x)[sigma1*(cos(x(1:nxnz)).*(Dxs*mx(x)) + sin(x(1:nxnz)).*(Dzs*mx(x)))
          sigma2*(-sin(x(1:nxnz)).*(Dxs*mx(x)) + cos(x(1:nxnz)).*(Dzs*mx(x)))];
Nabla = [Dx; Dz];

% bounds for upper level variable
l  = [-pi/2*ones(nxnz,1); 0]; 
u  = [pi/2*ones(nxnz,1); Inf];   

% options for L-BFGS-B (upper level solver) 
opts = struct('x0', [zeros(nxnz,1); 1000], 'factr', 1e4, 'pgtol', 1e-8, 'm', 10, 'maxIts', 100);
opts.printEvery     = 1;
if nxnz > 10000
    opts.m  = 50;
end

errFcn = cell(4,1);
errFcn{1} = @(x) x(end); % allows to print and store the value of the regularization parameter at each iteration
errFcn{2} = @(x) norm(mx(x)-X(:))/norm(X(:)); % allows to print and store the relative error at each iteration
errFcn{3} = @(x) norm(cos(x(1:end-1)).*(Dx*mx(x))+sin(x(1:end-1)).*(Dz*mx(x))); % allows to print and store the value of the first component of the directional derivative at each iteration
errFcn{4} = @(x) norm(-sin(x(1:end-1)).*(Dx*mx(x))+cos(x(1:end-1)).*(Dz*mx(x))); % allows to print and store the value of the second component of the directional derivative at each iteration
errFcn{5} = @(x) norm(mx(x) - Y(:));
errFcn{6} = @(x) norm(Ds(x));
errFcn{7} = @(x) norm(Nabla*(x(1:nxnz)));
opts.errFcn = errFcn;

% Run the algorithm:
[xk_pa, fk_pa, info_pa] = lbfgsb(fcn, l, u, opts );

gamma_pa = xk_pa;
D = [sigma1*(cos(gamma_pa(1:nxnz)).*Dx + sin(gamma_pa(1:nxnz)).*Dz)
     sigma2*(-sin(gamma_pa(1:nxnz)).*Dx + cos(gamma_pa(1:nxnz)).*Dz)];
coeffmatr = G'*G + gamma_pa(nxnz+1)*(D'*D);
m_pa = coeffmatr\(G'*Y(:));

%% compare with fully anisotropic Tikhonov with std derivative and reg parameter chosen according to the discrepancy principle
% NOTE: the ordering of the sigmas and thetas is swapped w.r.t. the paper. 
% I.e., gamma = [theta; sigmax; reg_parameter];
fcn = @(gamma)smoothulwithgrad(gamma, G, Y(:), epsilon, xi, Dx, Dz, Dxs, Dzs, nx, nz, beta2, beta1);
D = @(x)[x(nxnz+1:2*nxnz).*(cos(x(1:nxnz)).*Dx + sin(x(1:nxnz)).*Dz)
         (1 - x(nxnz+1:2*nxnz)).*(-sin(x(1:nxnz)).*Dx + cos(x(1:nxnz)).*Dz)];
coeffmatr = @(x)(G'*G + x(2*nxnz+1)*((D(x))'*(D(x))));
mx = @(x)coeffmatr(x)\(G'*Y(:));
Ds = @(x)[x(nxnz+1:2*nxnz).*(cos(x(1:nxnz)).*(Dxs*mx(x)) + sin(x(1:nxnz)).*(Dzs*mx(x)))
          (1 - x(nxnz+1:2*nxnz)).*(-sin(x(1:nxnz)).*(Dxs*mx(x)) + cos(x(1:nxnz)).*(Dzs*mx(x)))];

% bounds for upper level variable
l  = [-pi/2*ones(nx*nz,1); 0.5*ones(nxnz,1); 0];
u  = [pi/2*ones(nx*nz,1); ones(nx*nz,1); Inf];   

% options for L-BFGS-B (upper level solver)
opts = struct('x0', [gamma_pa(1:end-1); 0.9*ones(nxnz,1); gamma_pa(end)], 'factr', 1e4, 'pgtol', 1e-8, 'm', 10, 'maxIts', 100);
opts.printEvery     = 1;
if nxnz > 10000
    opts.m  = 50;
end

errFcn = cell(4,1);
errFcn{1} = @(x) x(end); % allows to print and store the value of the regularization parameter at each iteration
errFcn{2} = @(x) norm(mx(x)-X(:))/norm(X(:)); % allows to print and store the relative error at each iteration
errFcn{3} = @(x) norm(cos(x(1:nxnz)).*(Dx*mx(x))+sin(x(1:nxnz)).*(Dz*mx(x))); % allows to print and store the value of the first component of the directional derivative at each iteration
errFcn{4} = @(x) norm(-sin(x(1:nxnz)).*(Dx*mx(x))+cos(x(1:nxnz)).*(Dz*mx(x))); % allows to print and store the value of the second component of the directional derivative at each iteration
errFcn{5} = @(x) norm(mx(x) - Y(:));
errFcn{6} = @(x) norm(Ds(x));
errFcn{7} = @(x) norm(Nabla*(x(1:nxnz)));
errFcn{8} = @(x) norm(Nabla*(x(nxnz+1:2*nxnz)));
opts.errFcn = errFcn;

% Run the algorithm:
[xk, fk, info] = lbfgsb(fcn, l, u, opts );

gamma = xk;
D = [xk(nxnz+1:2*nxnz).*(cos(gamma(1:nxnz)).*Dx + sin(gamma(1:nxnz)).*Dz)
     (1-xk(nxnz+1:2*nxnz)).*(-sin(gamma(1:nxnz)).*Dx + cos(gamma(1:nxnz)).*Dz)];
coeffmatr = G'*G + gamma(2*nxnz+1)*(D'*D);
m = coeffmatr\(G'*Y(:));

%% Displaying the results
figure, subplot(2,2,1), imagesc(reshape(X(:), nz, nx)), axis image, axis off, title('exact')
subplot(2,2,2), imagesc(reshape(G'*Y(:), nz, nx)), axis image, axis off, title('corrupted')
subplot(2,2,3), imagesc(reshape(m_pa(:), nz, nx)), axis image, axis off, title('restored')
subplot(2,2,4), imagesc(reshape(gamma_pa(1:nxnz), nz, nx)), axis image, axis off, title('theta'), colorbar

figure, subplot(2,2,1), imagesc(reshape(X(:), nz, nx)), axis image, axis off, title('exact')
subplot(2,2,2), imagesc(reshape(gamma(nxnz+1:2*nxnz), nz, nx)), colorbar, axis image, axis off, title('\sigma_1')
subplot(2,2,3), imagesc(reshape(m(:), nz, nx)), axis image, axis off, title('restored')
subplot(2,2,4), imagesc(reshape(gamma(1:nxnz), nz, nx)), axis image, axis off, title('theta'), colorbar

figure, subplot(2,2,1), imagesc(reshape(m_DP(:), nz, nx)), axis image, axis off, title('restored, no direction')
subplot(2,2,2), imagesc((reshape(m_pa(:), nz, nx))), axis image, axis off, title('restored, partially anisotropic')
subplot(2,2,3), imagesc((reshape(m(:), nz, nx))), axis image, axis off, title('restored, fully anisotropic')

figure, subplot(2,1,1), semilogy(info_pa.err(:,1), 'o-'), title('objective fcn')
subplot(2,1,2), semilogy(info_pa.err(:,4), 's-r'), title('relative error')

figure, subplot(2,1,1), semilogy(info.err(:,1), 'o-'), title('objective fcn')
subplot(2,1,2), semilogy(info.err(:,4), 's-r'), title('relative error')