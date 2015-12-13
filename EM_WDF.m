function [obj] = EM_WDF(X,WS,K,varargin)
% [W,M,V,L,WS,X] = EM_WDF(X,WSampesAprior,K,Init)
%
% EM algorithm for weigthed data GMM  WITH fixed weight
%
% Inputs:
%   X(n,d) - input data, n=number of observations, d=dimension of variable
%   WS -input data weight
%   K - maximum number of Gaussian components allowed excluding out-liner
%   Init - structure of initial W, M, V: Init.W, Init.M, Init.V([] for none)
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

[Init,tol,maxIter,Regularize,CovType] = process_options(varargin, 'Init', [],'tol',1e-2,'maxIter',100,'Regularize',0,'CovType','full');
if(strcmp(CovType,'full'))
    covtype=2;
else % default is diag
    CovType='diag';
    covtype=1;
end

[n, d]=size(X);
%% Inititilize
% Initilize Mean, Cov  and component mixing weight by K-means, if not provided from as input
if isempty(Init),
    [Init.Wc,Init.M,Init.V] = EMInit(X,K,'kmeans');
end
% if provide in the input arg as a struct
W = Init.Wc;
M = Init.M;
V = Init.V;
%% Compute initial Expectation and Log likelihood
Ln = Likelihood(X,K,W,M,V,WS);
Lo = 2*Ln;
NlogL = -inf;
%% EM algorithm
niter = 1; L(niter)= Ln;
while ((niter<=maxIter) && (abs(100*(Ln-Lo)/Lo)>tol))
    [E,ll] = Expectation(X,K,W,M,V,WS,covtype); % E-step
    [W,M,V,K] = Maximization(X,K,E,WS,Regularize,covtype);  % M-step
    niter = niter + 1;
    L(niter)=ll;
    NlogL = ll;
end
% Store results in object
obj.DistName = 'GMM-WD Fixed';
obj.NDimensions = d;
obj.NComponents = K;
obj.PComponents = W;
obj.mu = M;
obj.Sigma = V;
obj.Wbar = WS;
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
else
    obj.CovType = 'full';
    nParam = K*obj.NDimensions * (obj.NDimensions+1)/2;
end
nParam = nParam + K-1 + K * obj.NDimensions;
obj.BIC = 2*NlogL + nParam*log(n);
obj.AIC = 2*NlogL + 2*nParam;
end

%% Expectation Step
function [E,ll] = Expectation(X,k,W,M,V,WS,covType)
[n,d] = size(X);
log_prior = log(W);
log_lh = zeros(n,k);
Wbar_nk = repmat(WS,1,k);
for i=1:k,
    if covType==2
        [L,err] = cholcov(V(:,:,i),0);
        if err ~= 0
            error(message('stats:mvnpdf:BadMatrixSigma'));
        end
        diagL = diag(L);
        logDetSigma = 2*sum(log(diagL));
    else
        L = sqrt(diag(V(:,:,i)));
        logDetSigma = sum(log(diag(V(:,:,i))));
    end
    dXM = bsxfun(@minus, X, M(:,i)');
    if covType == 2
        xRinv = dXM /L ;
    else
        xRinv = bsxfun(@times,dXM , (1./ L)');
    end
    tmp = Wbar_nk(:,i) .* sum(xRinv.^2, 2);
    log_lh(:,i) = -0.5 * tmp + (-0.5 *logDetSigma + log_prior(i)) - d*log(2*pi)/2 + d*log(Wbar_nk(:,i))/2;
end
maxll = max (log_lh,[],2);
%minus maxll to avoid underflow
post = exp(bsxfun(@minus, log_lh, maxll));
density = sum(post,2);
%normalize posteriors
E = bsxfun(@rdivide, post, density);
logpdf = log(density) + maxll;
ll = sum(logpdf) ;
end
% End of Expectation %

%% Maximization Step
function [W,M,V,k] = Maximization(X,k,E,WS,Regularize,covType)
[n,d] = size(X);
Wbar_nk = repmat(WS,1,k);
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
% End of Maximization %


function LL = Likelihood(X,k,W,M,V,WS)
[n,d] = size(X);
a = (2*pi)^(0.5*d);
LL = 0;
% variable to store the determinant of the covariances
S = zeros(1,k);
% variables to store the inverse convariances
iV = zeros(d,d,k);
% for each mixture component
for j=1:k,
    S(j) = sqrt(det(V(:,:,j)));
    iV(:,:,j) = inv(V(:,:,j));
end
% for each observation
%for each observations
for i=1:n
    p=0;
    for j=1:k
        if(W(j)>0)
            b = S(j)/(WS(i)+eps);
            % Mahalobis distanace
            dXM = X(i,:)'-M(:,j);
            MD = dXM'*(iV(:,:,j)/WS(i))*dXM;
            p = p + W(j)*exp(-0.5*MD)/(a*b);
        end
    end    
    LL= LL+ log(1-p);
   
end

end

