function obj = cWiseEM_WDMML(X,Kmax,varargin)
% CWiseEM_WDMML EM algorithm for weighted data clustering
%   obj = cWiseEM_WDMML(X,Kmax)
%   obj = cWiseEM_WDMML(X,Kmax,varargin)
%       X: [n x d] data matrix where n is the number of data points and d is the feature dimention
%       WSampesAprior: is [n x 1] a weight vector for data
%       Kmax: is the maximum number of component for component wise EM, Kmin=1 by default
%       varargin: varaible number of inputs
%           'Wdata' : weight of the data, if not given we compute from a kernel density
%           'init': type of initilization 1=(kmeans)[default],  2=(random)
%           'start': custom initilization start.W, start.M, start.V
%           'tol' = 1e-8 [default]
%           'Regularize'= 1e-6 [default]
%           'CovType'= 'full' [default]
%           'kmin' = 1 [default]
%           'fixcov' =1 [default]
%           'debg' = 0 [default]
%
%   see also EM_WD, EM_WDF




%#   $Author: Israel D. Gebru $    $Date: 2015/05/01 $    $Revision: 0.0 $
%#   Copyright:


[Wdata,init,start,tol,Regularize,CovType,kmin,fixcov,debg,makevideo] = process_options(varargin, 'Wdata',[],'init',2,'start',[],'tol',1e-8,'Regularize',1e-6,'CovType','full','kmin',1,'fixcov',1,'debg',0,'makevideo',0);
[n, d]=size(X);
if(strcmp(CovType,'full'))
    covtype=2;
    npars = d + d*(d-1)/2;
else % diagonal covaraince
    covtype=1;
    npars = 2*d ;
end

%% observation weights
if isempty(Wdata)
    % weight is proportional to the 2D kernel density
    % for high dim data project to 2D and compute the kerenel density
    Wdata = obWeights(X,'wtype',15);
else
    [n_wd,d_wd] = size(Wdata);
    if(n_wd~=n || d_wd>1)
        error('the size of the weight matrix do not match with the data size');
    end
end
alpha_n = Wdata;
alpha_nk = repmat(alpha_n+d/2,1,Kmax);
gamma_nk = repmat(sqrt(Wdata),1,Kmax);
alphaApriori = alpha_n;
gammaApriori = sqrt(Wdata);

%% Inititilize
% Init mean, varaince and component mixing weight by K-means
if isempty(start)
    init_type = {'kmeans','rand'};
    [W,M,V] = EMInit(X,Kmax,init_type{init});
else
    W = start.W;
    M = start.M;
    V = start.V;
    Kmax = length(W);
end

%% some varaibles we need
nparsover2 = npars / 2;
k = Kmax;
% using the initial means, covariances, and probabilities, to compute the log-likelihood
[~,ll,~,~] = Expectation(X,Kmax,W,M,V,alpha_nk,gamma_nk,alphaApriori,gammaApriori,covtype); % E-step
niter = 1;
loglike(niter) =  -inf;%2*ll; % stores the log-likelihood
dlength = -loglike(niter) + (nparsover2*sum(log(W))) + (nparsover2 + 0.5)*k*log(n); % description length
dl(niter) = dlength;  % stores the description length
kappas(niter) = k;  %  store the number of components
% the transitions vectors will store the iteration number at which components are killed.
% transitions1 stores the iterations at which components are  killed by the M-step,
% while transitions2 stores the iterations at which we force components to zero.
transitions1 = [];
transitions2 = [];

% minimum description length seen so far, and corresponding % parameter estimates
mindl = dl(niter);
bestW = W;
bestM = M;
bestV = V;
bestK = k;
bestalpha_nk= alpha_nk;
bestgamma_nk = gamma_nk;
kappas=[];
k_cont = 1;    % auxiliary variable for the outer loop

%% debug
if(makevideo)
    figh = figure('Position',[-1508 1003 1278 872],'PaperOrientation', 'portrait','Visible','on');
    pl1 = subplot(2,3,1);
    plotcluster(X,ones(n,1)+4,10+1,'WD-EM-MML INIT',pl1);hold on;
    for j=1:k
        Plot_Std_Ellipse(M(:,j),V(:,:,j),gca,3); hold on; axis off;
    end
    set(gca, 'LooseInset', [0,0,0,0]);
    pl3 = subplot(2,3,3);cla;
    plotcluster(X,ones(n,1)+4,10+1,'Best',pl3);hold on;
    clrmstep = 1;
    for j=1:k
        Plot_Std_Ellipse(bestM(:,j),bestV(:,:,j),gca,clrmstep); hold on; axis off;
    end
    set(gca, 'LooseInset', [0,0,0,0]);
    
    subplot(2,3,4);cla;
    plot(1:niter,loglike,'LineStyle','-','linewidth',2,'Marker','o','MarkerSize',6,'MarkerEdgeColor','none');
    set(gca,'YLim',[loglike(1),0],'XLim',[1,200]);
    subplot(2,3,5:6);cla;
    plot(1:niter,dl,'LineStyle','-','linewidth',2,'Marker','o','MarkerSize',6,'MarkerEdgeColor','none');
    set(gca,'YLim',[0,dl(1)],'XLim',[1,500]);
    mframe(1) = getframe(gcf);
end
while(k_cont)  % the outer loop will take us down from kmax to kmin components
    cont=1;        % auxiliary variable of the inner loop
    while(cont)    % this inner loop is the component-wise EM algorithm with the
        prt(debg,2,sprintf('k = %2d,  minestpp = %0.5g @iter=', k, min(W)),niter);
        
        % we begin at component 1
        comp = 1;
        while comp <= k
            [E,~,alpha_nk,gamma_nk] = Expectation(X,k,W,M,V,alpha_nk,gamma_nk,alphaApriori,gammaApriori,covtype); % E-step
            %% now we perform the standard M-step for Mean and Covariance
            Wbar_nk =  alpha_nk./gamma_nk;
            M = zeros(d,k); V = zeros(d,d,k);
            for j=1:k
                den = Wbar_nk(:,j).*E(:,j);
                tmp = den'*X;
                M(:,j) = tmp'/sum(den,1);
                dxM = bsxfun(@minus,X,M(:,j)');
                if(covtype ==2) % full cov
                    dxM = bsxfun(@times,sqrt(E(:,j).*Wbar_nk(:,j)),dxM);
                    V(:,:,j) = V(:,:,j) + dxM'*dxM;
                else  % diagonal cov
                    V(:,:,j) = V(:,:,j) + diag(den'*(dxM.^2));
                end
                % normalize the new covariance + regularzation
                V(:,:,j) = V(:,:,j)/sum(E(:,j),1)+ eye(d)*(Regularize);
                if fixcov
                    [V(:,:,j),loops] = covfixer(V(:,:,j));
                    % covfixer may change the matrix so that log-likelihood
                    if(loops>5) % 5
                        error('tried hard to fix the cov, but this it too much to fix!')
                    end
                end
            end
            
            % this is the special part of the M step that is able to
            % kill components
            W(comp) = max(sum(E(:,comp))-nparsover2,0)/n;
            W = W/sum(W);
            % this is an auxiliary variable that will be used to signal the killing of the current component being updated
            killed = 0;
            % do some book-keeping if the current component was killed
            if W(comp)==0
                if(makevideo)
                    pl2 = subplot(2,3,2);cla;
                    plotcluster(X,ones(n,1)+4,10+1,'CEM-Step',pl2);hold on;
                    clrmstep = 2;
                    for j=1:k
                        if(comp==j)
                            clrmstep = 5;
                        end
                        Plot_Std_Ellipse(M(:,j),V(:,:,j),gca,clrmstep); hold on; axis off;
                    end
                    set(gca, 'LooseInset', [0,0,0,0]);
                    mframe(end+1) = getframe(gcf);
                end               
                
                prt(debg,2,'component killed..',comp);
                killed = 1;
                % we also register that at the current iteration a component was killed
                transitions1 = [transitions1,niter];
                V(:,:,comp) = [];
                M(:,comp) = [];
                W(comp) = [];
                Wbar_nk(:,comp) = [];
                gamma_nk(:,comp) = [];
                alpha_nk(:,comp) = [];
                % since we've just killed a component, k must decrease
                k=k-1;
            end % end of W(comp)==0
            
            % if the component was not killed
            if killed==0
                comp = comp + 1;
            end
            % if killed==1, it means the in the position "comp", we now have a component that was not yet visited in this sweep,
            % and so all we have to do is go back to the M setp without increasing "comp"
            
        end % this is the end of the innermost "while comp <= k" loop which cycles through the components
        
        % increment the iterations counter
        niter = niter + 1;
        %perform E-step
        [~,ll,~,~] = Expectation(X,k,W,M,V,alpha_nk,gamma_nk,alphaApriori,gammaApriori,covtype); % E-step
        loglike(niter) = ll;
        % compute and store the description length and the current number of components
        dlength = -loglike(niter) + (nparsover2*sum(log(W))) +  (nparsover2 + 0.5)*k*log(n);
        dl(niter) = dlength;
        kappas(niter) = k;
        % compute the change in loglikelihood to check if we should stop
        deltlike = loglike(niter) - loglike(niter-1);
        prt(debg,1,sprintf('########### iter: %d, deltaloglike = %0.12f%% ,K=%d, BestK=',niter,abs(100*(deltlike/loglike(niter-1))),k),bestK);
        if abs(100*(deltlike/loglike(niter-1))) < tol
            % if the relative change in loglikelihood is below the tolerance threshold, we stop
            cont=0;
        end
        if(makevideo)
            pl2 = subplot(2,3,2);cla;
            plotcluster(X,ones(n,1)+4,10+1,'CEM-Step',pl2);hold on;
            clrmstep = 2;
            for j=1:k
                if(comp==j)
                    clrmstep = 4;
                end
                Plot_Std_Ellipse(M(:,j),V(:,:,j),gca,clrmstep); hold on; axis off;
            end
            set(gca, 'LooseInset', [0,0,0,0]);
            subplot(2,3,4);cla;
            plot(1:niter,loglike,'LineStyle','-','linewidth',2,'Marker','o','MarkerSize',6,'MarkerEdgeColor','none');
            set(gca,'YLim',[loglike(1),0],'XLim',[1,200]);
            subplot(2,3,5:6);cla;
            plot(1:niter,dl,'LineStyle','-','linewidth',2,'Marker','o','MarkerSize',6,'MarkerEdgeColor','none');
            set(gca,'YLim',[0,dl(1)],'XLim',[1,500]);
            mframe(end+1) = getframe(gcf);
        end
    end % this end is of the inner loop: "while(cont)"
    
    % now check if the latest description length is the best if it is, we store its value and the corresponding estimates
    if dl(niter) < mindl
        bestW = W;
        bestM = M;
        bestV = V;
        bestK = k;
        bestalpha_nk= alpha_nk;
        bestgamma_nk = gamma_nk;
        mindl = dl(niter);
    end
    
    %% random reshuffling
    % the order of updating does not affect the theoretical monotonicity properties of CEM
    % but I like it random ,so just to randomized stuffs
    %     randindex = randperm(k,k);
    %     W = W(randindex);
    %     M = M(:,randindex);
    %     V = V(:,:,randindex);
    %     alpha_nk = alpha_nk(:,randindex);
    %     gamma_nk = gamma_nk(:,randindex);
    
    % at this point, we may try smaller mixtures by killing the component with the smallest mixing probability
    %and then restarting CEM2 as long as k is not yet at kmin
    if k>kmin
        [~, indminw] = min(W);
        V(:,:,indminw) = [];
        M(:,indminw) = [];
        W(indminw) = [];
        Wbar_nk(:,indminw) = [];
        gamma_nk(:,indminw) = [];
        alpha_nk(:,indminw) = [];
        k = k-1;
        % we renormalize the mixing probabilities after killing the component
        W = W./sum(W);
        % and register the fact that we have forced one component to zero
        transitions2 = [transitions2, niter];
        % increment the iterations counter
        niter = niter+1;
        % compute the loglikelihhod function and the description length
        [~,ll,~,~] = Expectation(X,k,W,M,V,alpha_nk,gamma_nk,alphaApriori,gammaApriori,covtype); % E-step
        loglike(niter) = ll;
        dlength = -loglike(niter) + (nparsover2*sum(log(W))) +  (nparsover2 + 0.5)*k*log(n);
        dl(niter) = dlength;
        kappas(niter) = k;
    else
        %if k is not larger than kmin, we must stop
        k_cont = 0;
    end
    if(makevideo)
        pl3 = subplot(2,3,3);cla;
        plotcluster(X,ones(n,1)+4,10+1,'Best',pl3);hold on;
        clrmstep = 1;
        for j=1:bestK
            Plot_Std_Ellipse(bestM(:,j),bestV(:,:,j),gca,clrmstep); hold on; axis off;
        end
        set(gca, 'LooseInset', [0,0,0,0]);
        
        subplot(2,3,4);cla;
        plot(1:niter,loglike,'LineWidth',2,'linesmoothing','on');
        set(gca,'YLim',[loglike(1),0],'XLim',[1,200]);
        subplot(2,3,5:6);cla;
        plot(1:niter,dl,'LineStyle','-','linewidth',2,'Marker','o','MarkerSize',6,'MarkerEdgeColor','none');
        set(gca,'YLim',[0,dl(1)],'XLim',[1,500]);
        mframe(end+1) = getframe(gcf);
    end
end % this is the end of the outer loop "while(k_cont)"
[E,~,~,~] = Expectation(X,bestK,bestW,bestM,bestV,bestalpha_nk,bestgamma_nk,alphaApriori,gammaApriori,covtype); % E-step

%%
% to merge exactly similar gaussian. This happen when we have points with strong weight concentrated in one area
% the algorithm will fail to annihilate one of them, thus we do it here
mu = bestM;
D= inf(bestK,bestK);
D_zero =zeros(bestK,bestK);
for i=1:bestK
    for j=i+1:bestK
        D(i,j) = pdist2(mu(:,i)',mu(:,j)');
    end
end
[r,c,~] = find(D==D_zero);
%remove one of them
if ~isempty(r)
    prt(debg,1,'Identical comp, removing one of them!',1);
    bestK=bestK-1;
    bestW(r)=[];
    bestM(:,r)=[];
    bestV(:,:,r)=[];
    bestalpha_nk(:,r) = [];
    bestgamma_nk(:,r) = [];
    % need to recompute
    [E,ll,~,~] = Expectation(X,bestK,bestW,bestM,bestV,bestalpha_nk,bestgamma_nk,alphaApriori,gammaApriori,covtype);
    mindl = -ll + (nparsover2*sum(log(bestW))) +  (nparsover2 + 0.5)*bestK*log(n);
end
if(makevideo)
    pl3 = subplot(2,3,3);cla;
    plotcluster(X,ones(n,1)+4,10+1,'Best',pl3);hold on;
    clrmstep = 1;
    for j=1:bestK
        Plot_Std_Ellipse(bestM(:,j),bestV(:,:,j),gca,clrmstep); hold on; axis off;
    end
    set(gca, 'LooseInset', [0,0,0,0]);
    
    subplot(2,3,4);cla;
    plot(1:niter,loglike,'LineWidth',2,'linesmoothing','on');
    set(gca,'YLim',[loglike(1),0],'XLim',[1,200]);
    subplot(2,3,5:6);cla;
    plot(1:niter,dl,'LineStyle','-','linewidth',2,'Marker','o','MarkerSize',6,'MarkerEdgeColor','none');
    set(gca,'YLim',[0,dl(1)],'XLim',[1,500]);
    mframe(end+1) = getframe(gcf);
end
%%
% Store results in object
obj.Iters = niter;
obj.DistName = 'GMM-WD with component wise EM algorithm';
obj.NDimensions = d;
obj.NComponents = bestK;
obj.PComponents = bestW;
obj.mu = bestM;
obj.Sigma = bestV;
Wbar_nk = bestalpha_nk./bestgamma_nk;
obj.Wbar = sum(Wbar_nk.*E,2);
obj.E = E;
[~, idx] = max(E,[],2);
obj.Class =  idx;
obj.Iters = niter;
obj.logL =  loglike;
obj.RegV = Regularize;
obj.kappas = kappas;
obj.dl = dl;
obj.mindl = mindl;
if makevideo
    obj.fMovie = mframe;
end
end

%% Expectation Step
function [E,ll,alpha_n_new,gamma_nk_new] = Expectation(X,k,W,M,V,alpha_nk,gamma_nk,alphaApriori,gammaApriori,covType)
[n,d] = size(X);
log_prior = log(W);
log_lh = zeros(n,k);
mahalaD = zeros(n,k);
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

function [pdf] = mvpearsonType7pdf(x,mu,sigma,alpha,beta) % alpha is really alpha + d/2
[~,d] = size(x);
dover2 = d/2;
[L,er] = chol(sigma);
if(er~=0)
    error(message('stats:gmdistribution:wdensity:IllCondCov'));
end
sqrtdetSigma = det(L); %  sigma = L'L, det(sigma)=det(L')det(L), sqr(det(L))
dXM = bsxfun(@minus, x, mu); % centering
xRinv = dXM/L ;
Mahal_over2beta = sum(xRinv.^2, 2)./(2*beta);
denm = sqrtdetSigma.*gamma(alpha-dover2).* (2*pi.*beta).^(dover2) .* (1 +Mahal_over2beta).^(-alpha);
pdf  = gamma(alpha)./denm;
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

