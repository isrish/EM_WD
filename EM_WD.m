function obj = EM_WD(X,K,varargin)
% [obj] = EM_WD(X,K,varargin)
%
% EM algorithm for weigthed data clustering
%
% Inputs:
%   X(n,d) - input data, n=number of observations, d=dimension of variable
%   K - maximum number of Gaussian components allowed excluding out-liner
%   varargin
%       'Init' - struct of initial W, M, V: Init.W, Init.M, Init.V([] for none)
%       'Wdata' : is [n x 1] a weight vector for data, if not given we compute from a kernel density    
%       'tol' -  change of log-likelihood to convergence, default is 0.01 %
%       'maxIter' - number of max iterations to run EM, default 100 iterations.
%       'Regularize' - A nonnegative regularization number added to the diagonal of covariance matrices to make them
%       psd, default 0.
%       'CovType' - covariance matrix, 'full' or 'diag'
%
% Ouputs: Obj: a struct with the following elements
%   PComponents(1,k) - estimated weights of mixture compontents
%   mu(d,k) - estimated mean vectors
%   Sigma(d,d,k) - estimated covariance matrices
%   L - log likelihood of estimates
%   Wbar - expecation on the weights after EM convergence
%   E(N,k)  - posterior
%   Class -  MAP class labels
%   Iters - number of iterations till convergence
%   NlogL -  negative of log likelihood
%   BIC -  Bayesian information criterion (BIC) or Schwarz criterion
%   AIC -  Akaike information criterion 

% $Author: Israel D. Gebru $    $Date: Dec 1, 2014$    $Revision: 1.0 $
%                               $Date: May 1, 2015$    $Revision: 2.0 $    
% Copyright: Perception Team, INRIA-Grenoble
% Email: israel--dejene.gebru@inria.fr

% The EM algorithm is described in the paper:
%  (1) Israel D.,Xavier A., Florence F., Radu H., "EM for Weighted-Data Clustering" ,
%       ~~
%  (2) Israel D.,Xavier A., Radu H.,Florence F.,  "Audio-visual speaker localization via weighted clustering",
%      IEEE International Workshop onMachine Learning for Signal Processing (MLSP), 2014.

[Init,Wdata,tol,maxIter,Regularize,CovType,debg] = process_options(varargin, 'Init', [],'Wdata',[],'tol',1e-2,'maxIter',100,'Regularize',1e-6,'CovType','diag','debg',0);
if(strcmp(CovType,'full'))
    covtype=2;
else % default is diag
    CovType='diag';
    covtype=1;
end
[n, d]=size(X);

%% observation weights
if isempty(Wdata)
    % weight is proportional to the 2D kernel density
    % for high dim data project to 2D and compute weight from 2D kernel density
    Wdata = obWeights(X,'wtype',15);
else
    [n_wd,d_wd] = size(Wdata);
    if(n_wd~=n || d_wd>1)
        error('the size of the weight matrix do not match with the data size');
    end
end
alpha_n = Wdata;
alpha_nk = repmat(alpha_n+d/2,1,K);
gamma_nk = repmat(sqrt(Wdata),1,K);
alphaApriori = alpha_n;
gammaApriori = sqrt(Wdata);
Wbar_nk= repmat(alpha_n,1,K)./gamma_nk;

%% Inititilize
% Initilize Mean, Cov  and component mixing weight by K-means, if not provided from as input
if isempty(Init),
    [Init.Wc,Init.M,Init.V] = EMInit(X,K,'kmeans');
end
% if provide in the input arg as a struct
W = Init.Wc;
M = Init.M;
V = Init.V;

%% EM algorithm
Lo = -inf;
NlogL = -inf;
niter = 1; L(niter)= Lo;
obj.Converged = false;
obj.Iters=0;
while true
    [E,Ln,alpha_nk,gamma_nk] = Expectation(X,K,W,M,V,alpha_nk,gamma_nk,alphaApriori,gammaApriori,covtype); % E-step
    Wbar_nk = alpha_nk./gamma_nk;
    [W,M,V,K] = Maximization(X,K,E,Wbar_nk,Regularize,covtype);  % M-step
    prt(debg, 1, ['########### ' num2str(niter) ',nll='],-Ln);
    Lo = L(niter);
    NlogL = -Ln;
    niter = niter + 1;
    L(niter)= -Ln;
    obj.Iters = niter;
    if(niter>maxIter)
        break;
    end
    if abs(100*(Ln-Lo)/Lo)<=tol && niter<=maxIter
        obj.converge = true;
        break;
    end
    
end
% Store results in object
obj.DistName = 'GMM-WD';
obj.NDimensions = d;
obj.NComponents = K;
obj.PComponents = W;
obj.mu = M;
obj.Sigma = V;
obj.Wbar = sum(Wbar_nk.*E,2);
obj.E = E;
[~, idx] = max(E,[],2);
obj.Class =  idx;
obj.Iters = niter;
obj.NlogL =  NlogL;
obj.L = L;
obj.RegV = Regularize;
if covtype == 1
    obj.CovType = 'diag';
    nParam = obj.NDimensions * K;
elseif covtype==2
    obj.CovType = 'full';
    nParam = K*obj.NDimensions * (obj.NDimensions+1)/2;
end
nParam = nParam + K * obj.NDimensions + K-1; % #covar, #mean, #pi 
obj.BIC = 2*NlogL + nParam*log(n); % best K is that min BIC
obj.AIC = 2*NlogL + 2*nParam;

objGMM = gmdistribution(M',V,W);
[~,nlogl] = posterior(objGMM,X);
obj.gmmBIC = 2*nlogl + nParam*log(n); 
obj.gmmAIC = 2*nlogl + 2*nParam;
end

%% Expectation Step
function [E,ll,alpha_n_new,gamma_nk_new] = Expectation(X,k,W,M,V,alpha_nk,gamma_nk,alphaApriori,gammaApriori,covType)
[n,d] = size(X);
log_prior = log(W);
log_lh = zeros(n,k);
mahalaD = zeros(n,k);
logDetSigma = -Inf;
wbar_nk = alpha_nk./gamma_nk;
%% E-Z step
for i=1:k,
    if covType==2 % full covariance
        [L,err] = chol(V(:,:,i));
        diagL = diag(L);
        if err ~= 0 || any(abs(diagL) < eps(max(abs(diagL)))*size(L,1))
            error(message('stats:gmdistribution:wdensity:IllCondCov'));
        end
        logDetSigma = 2*sum(log(diagL));
    else %diagonal
        L = sqrt(diag(V(:,:,i)));
        if  any(L < eps(max(L))*d)
            error(message('stats:gmdistribution:wdensity:IllCondCov'));
        end
        logDetSigma = sum(log(diag(V(:,:,i))));
     end
    dXM = bsxfun(@minus, X, M(:,i)'); % centering
    if covType == 2
        xRinv = dXM/L ;
    else
        xRinv = bsxfun(@times,dXM , (1./ L)');
    end
    mahalaD(:,i) = sum(xRinv.^2, 2);
    log_lh(:,i) = log_prior(i)+ gammaln(alpha_nk(:,i)) - 0.5 *logDetSigma - gammaln(alpha_nk(:,i)-d/2) - d/2*log(2*pi*gamma_nk(:,i))...
        -alpha_nk(:,i) .* log( 1 + 0.5 * mahalaD(:,i)./gamma_nk(:,i));
    %log_lh(:,i) = -0.5 * mahalaD(:,i) + (-0.5 *logDetSigma + log_prior(i)) - d*log(2*pi)/2;
end
maxll = max (log_lh,[],2);
%minus maxll to avoid underflow
post = exp(bsxfun(@minus, log_lh, maxll));
density = sum(post,2);
%normalize posteriors
E = bsxfun(@rdivide, post, density);
logpdf = log(density) + maxll;
ll = sum(logpdf) ;

%% E-W step
gamma_nk_new = zeros(n,k);
for i=1:k
    gamma_nk_new(:,i)= gammaApriori + 0.5* mahalaD(:,i);
end
alpha_n_new = repmat(alphaApriori + d/2,1,k);
end


%% Maximization Step
function [W,M,V,k] = Maximization(X,k,E,Wbar_nk,Regularize,covType)
%%% Auxiliar variables
% n ~ number of points
% d ~ data dimension
% W ~ new component mixing weight including uniform component
% M ~ new component mean
% V ~ new component varaince
[n,d] = size(X);
M = zeros(d,k); V = zeros(d,d,k);
%for each component Compute new mean, covar and componet weight
for i=1:k,
    den = Wbar_nk(:,i).*E(:,i);
    tmp = den'*X;
    M(:,i) = tmp'/sum(den,1);
    dxM = bsxfun(@minus,X,M(:,i)');
    if(covType ==2) % full cov
        dxM = bsxfun(@times,sqrt(E(:,i).*Wbar_nk(:,i)),dxM);
        V(:,:,i) = V(:,:,i) + dxM'*dxM;
    else  % diagonal
        V(:,:,i) = V(:,:,i) + diag(den'*(dxM.^2));
    end
    %normalize the new covariance + regularzation
    V(:,:,i) = V(:,:,i)/sum(E(:,i),1)+  eye(d)*(Regularize);
end
% mixing weights
W = sum(E,1)/n;
end

%% Log Likelihood
function LL = Likelihood(X,k,W,M,V,alpha_n,gamma_nk,Init)
[n,d] = size(X);
LL = 0;
% variable to store the determinant of the covariances
S = zeros(1,k);
% variables to store the inverse convariances
iV = zeros(d,d,k);
% for each mixture component
for j=1:k,
    % Make sure covariance matrix is a valid covariance matrix (PSD)
    [~,U,~] = lu(V(:,:,j));
    dt = prod(diag(U));
    if(dt<=1e-3), V(:,:,j) =  Init.V(:,:,j) + eye(d)*(1e-3);end  % reconstruct covariance matrix if not "valid" covariances matrix
    % Compute the sqrt of the determinant
    S(j) = sqrt(det(V(:,:,j)));
    % Invert the matrix
    iV(:,:,j) = inv(V(:,:,j));
end
% for each observation , TODO=vectorize this??
for i=1:n
    al= alpha_n(i,1)+d/2;
    ga= gamma_nk(i,:);
    bnk=0;
    for j=1:k
        bn =  S(j) * gamma(alpha_n(i,1))* (2*pi*ga(j))^(d/2);
        dXM = X(i,:)'-M(:,j);
        MD = dXM'*iV(:,:,j)*dXM; % sqr Mahalobis distanace
        bnk = bnk + 1/bn *(1 + 0.5*MD/ga(j))^(-al) * W(j);
    end
    LL = LL + log(k) + gammaln(al) + log(bnk);
end
% approximation to prevent numberical problem on high dim. data
if(d>50)
    md = zeros(n,k);  % Preallocate matrix
    for j=1:k
        md(:,j) = sqrt(mahaldist(X,M(:,j)',iV(:,:,j)));
    end
    a = zeros(n,k);  % Preallocate matrix
    [~, kidx] = min(md,[],2);
    for i=1:n
        al =alpha_n(i,1)+d/2;
        ga =gamma_nk(i,kidx(i));
        a(i,kidx(i)) = gammaln(al) - gammaln(alpha_n(i,1))- (d/2)*log(2*pi*ga) -al *log(1 + 0.5*md(i,kidx(i))/ga);
    end
    LL = sum(sum(a));
end
end

function prt(debg, level, txt, num)
% Print text and number to screen if debug is enabled.
if(debg >= level)
    if(numel(num) == 1)
        disp([txt num2str(num)]);
    else
        disp(txt)
        disp(num)
    end
end
end

