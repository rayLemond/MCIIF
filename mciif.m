function [P_hat, T] = mciif(P, lambda1, lambda2, opts)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% P: transition probability matrix, n x n x v.
% n: num of data. v: num of view.
% lambda1, lambda2: model parameters in equation.1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% some model-irrelevant settings (ADMM parameters)
if isfield(opts, 'max_iter')
    max_iter = opts.max_iter;
else
    max_iter = 500;
end

if isfield(opts, 'eta')
    eta = opts.eta;
else
    eta = 2.01;
end

if isfield(opts, 'beta')   
    beta = opts.beta;
else
    beta = 0.0002;
end

if isfield(opts, 'rho')
    rho = opts.rho;
else
    rho = 1.1;
end

%% initialization
[n, ~, v] = size(P);  
P_hat = zeros(n, n);     % Inter-view Graph, the shared low rank transition
Q = zeros(n, n);         % The auxiliary variable Q in equation.4 (a shadow P)
Z = zeros(n, n);         % Difference of (P,Q) in equation.4
E = randn(n, n, v);       % L1 noise for each view
E_old = E;
T = zeros(n, n, v);       % Low rank Intra-view Difference Graph for each view
T_old = T;
Y = zeros(n, n, v);       % linear constraint variable in Lagrange multipliers
P_old = randn(n, n);

%% start training
fprintf('starting training MCIIF ...\n');
step = 0;
while(1)

    step = step + 1;
    
    %% surveillance for training
    
    % max infinite norm of all views
    max_inf_norm = -1;
    for i = 1 : v  
        diff = P(:,:,i) - T(:,:,i) - P_hat;
        inf_norm = norm(diff, 'inf');
        max_inf_norm = max(max_inf_norm, inf_norm);
    end

    % loss function value
    lossfuncV = sum(svd(P_hat)); % nuclear norm value of P_hat
    for i = 1 : v
        lossfuncV = lossfuncV + lambda1*(sum(svd(T(:,:,i))) + lambda1*lambda2*norm(E(:,:,i), 1));
    end
    if mod(step, 50)==0 || step == max_iter-1
        fprintf('ite %d loss:%f  \n',step, lossfuncV);
    end
        
    % change degree of P_hat 
    relChg = norm(P_hat-P_old, 'fro')/max(1, norm( P_old, 'fro' ));
    P_old = P_hat;
    tmp = P_hat - Q;
    max_inf_norm2 = norm(tmp(:),'inf');

    % show dubug info or not
    if opts.DEBUG
        fprintf( 'iter %d: max_inf_norm = %s, relChg=%s, inf_norm2=%s, funV=%f\n',...
            step, num2str(max_inf_norm),num2str(relChg), num2str(max_inf_norm2),lossfuncV);
    end

    if step>1 && abs(lossfuncV-lossfuncV_old)<opts.eps
        break;
    end
    lossfuncV_old = lossfuncV;

    if step > max_iter
         fprintf('reach max iterations %d \n',step-1);
         break; 
    end

    %% update P_hat
    C = 1/(v+1)*(Q-Z/beta-sum(T+E-P+Y/beta,3)); % see equation.6
%         [C_m,~] = size(C);
%         for i = 1 : C_m
%            P(i,:) = projOnSimplex(C(i,:),1);
%         end
    ASC_opts.tol = 1e-5;
    ASC_opts.DEBUG = 0;
    P_hat = nonnegASC(C);   % see equation.7
    for i = 1 : n
        if sum(P_hat(i,:))-1.0 >= 1e-10
            error('sum to 1 error');
        end
    end
%         fprintf('P_hat:%d ', sum(svd(P_hat)));    % show nuclear norm value

    %% update Q 
    tempM1 = P_hat + Z/beta;    % see equation.9
    tempPara1 = 1/beta;
    [U, Sigma1, V] = svd(tempM1, 'econ');
    Sigma1 = diag(Sigma1);
    svp1 = length(find(Sigma1>tempPara1));
    if svp1 >= 1           
        Sigma1 = Sigma1(1:svp1) - tempPara1;
    else
        svp1 = 1;
        Sigma1 = 0;
    end
    Q = U(:, 1:svp1) * diag(Sigma1) * V(:, 1:svp1)';
%         fprintf('Q:%d  ',sum(svd(Q)));    % show nuclear norm value

    %% alternate Yv, which is Y(:,:,i), see equation.10
    for i = 1 : v
       Y(:,:,i) = Y(:,:,i) - beta*(P_hat+T(:,:,i)+E(:,:,i)-P(:,:,i)); 
    end

    %% update T
    for i = 1 : v % see equation.12
       tempM2 = T(:,:,i) + 1/(eta*beta)*Y(:,:,i);   
       tempPara2 = lambda1/(beta*eta*v);
       [U, Sigma2, V] = svd(tempM2, 'econ');
       Sigma2 = diag(Sigma2);
       svp2 = length(find(Sigma2>tempPara2));
       if svp2 >= 1
          Sigma2 = Sigma2(1:svp2) - tempPara2; 
       else
           svp2 = 1;
           Sigma2 = 0;
       end
       T(:,:,i) = U(:, 1:svp2) * diag(Sigma2) * V(:, 1:svp2)';
%            fprintf('T(%d):%d  ', i, sum(svd(T(:,:,i))));
    end

    %% update E
    for i = 1 : v   % see equation.11
        tempC = E(:,:,i)+1/(eta*beta)*Y(:,:,i);
        temppara = lambda1*lambda2/(beta*eta*v);
        E(:,:,i) = max(tempC-temppara, 0) + min(tempC+temppara, 0);
%             fprintf('E(%d):%d  ', i, norm(E(:,:,i), 1));
    end

    %% update Yv, which is Y(:,:,i), see equation.13
    for i = 1 : v
       Y(:,:,i) = Y(:,:,i) + beta*(T_old(:,:,i)-T(:,:,i)) + beta*(E_old(:,:,i)-E(:,:,i));
%        fprintf(' %06.4f',norm(Y(:,:,i),'fro'));
    end
    
    %%
    T_old = T;
    E_old = E;
    
    %% update Z
    Z = Z + beta * (P_hat-Q);

    %% update beta
    beta = min(rho*beta, 1e10);

end   % end for the loop
    
%% normalization if needed
% [pi,~]=eigs(P_hat', 1);
% Dist=pi/sum(pi);
% pi=diag(Dist);
% % P=(pi*P+P'*pi)/2;
% P_hat=(pi^0.5*P_hat*pi^-0.5+pi^-0.5*P_hat'*pi^0.5)/2;
