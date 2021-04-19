% load('dts_WikipediaArticles.mat')
load('dts_bbc4view.mat')
addpath('./code_coregspectral');

% initialization
opts.DEBUG = 0;     % 1: show detailed training info
opts.eps = 1e-6;
numView = size(data,2);

% construct transition matrix
K=[];
P=[];
for j = 1 : numView
    options.KernelType = 'Gaussian';
    options.t = 100;                 
    K(:,:,j) = constructKernel(data{j}, data{j}, options);
    D = diag(sum(K(:,:,j), 2));
    L_rw = D^-1 * K(:,:,j);
    P(:,:,j) = L_rw;
end

% run MCIIF
lambda1 = 1e-3; % see section 4.4 for parameter sensitivity
lambda2 = 1e4;
[P_hat, T] = mciif(P, lambda1, lambda2, opts);  % P_hat: Inter-view low rank graph
for t = 1 : numView                             % T: Intra-view low rank graph
    P_hat = P_hat + 1/numView*T(:,:,t);
end  

% evaluation using source code of (Coregularized multiview spectral clustering, NIPS2011)
numClust = length(unique(labels));
projev = 1.5;  % top (projev*numClust) eigenvectors of the Laplacian
[~,~,F,P,R,nmi,avgent,AR,~,~,prty] = baseline_spectral_onRW(P_hat, numClust, labels, projev); 
fprintf(' F-score=%6f Purity=%6f NMI=%6f ARI=%6f Entropy=%6f\n', F(1), prty(1), nmi(1), AR(1), avgent(1));