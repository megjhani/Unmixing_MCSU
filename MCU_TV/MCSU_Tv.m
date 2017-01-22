function [ X_hat,res,rmse_ni] = MCSU_Tv( M,Y,varargin )
%MCSU_TV Summary of this function goes here
%   min  (1/2) lambda_d||A X-Y||^2_F  + lambda_1  ||X||_{1,1}
%    D,G,X                         + lambda_tv ||LX||_{1,1}
%                              + ||DG-PS||^2_F + lamda_g||G||_1;
% Step 1 : Dictionary Learning Stage and Sparse Coding
%         min ||DG-PS||^2_F + lamda_g||G||_1;
%          D,G
% Step 2 : Mixture Learning Stage
%   min  (1/2) ||A' X-Y'||^2_F  + lambda_1  ||X||_{1,1}
%    X                         + lambda_tv ||LX||_{1,1}
% Where A' = [sqrt(lambda_d)*A^T I^T]^T
%       Y' = [sqrt(sqrt(lambda_d)*Y^T (pinv(P)DG)^T]^T




%%
%--------------------------------------------------------------
% test for number of required parametres
%--------------------------------------------------------------
if (nargin-length(varargin)) ~= 2
    error('Wrong number of required parameters');
end
% mixing matrix size
[LM,n] = size(M);
% data set size
[L,N] = size(Y);
if (LM ~= L)
    error('mixing matrix M and data set y are inconsistent');
end

lamda_d = 0.1;
w_r = 15;
sizeOfDictionary = 500*size(Y,1);
if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'LAMBDA_1'
                lambda_l1 = varargin{i+1};
                if lambda_l1 < 0
                    error('lambda must be positive');
                elseif lambda_l1 > 0
                    reg_l1 = 1;
                end
            case 'LAMBDA_TV'
                lambda_TV = varargin{i+1};
                if lambda_TV < 0
                    error('lambda must be non-negative');
                elseif lambda_TV > 0
                    reg_TV = 1;
                end
            case 'TV_TYPE'
                tv_type = varargin{i+1};
                if ~(strcmp(tv_type,'iso') | strcmp(tv_type,'niso'))
                    error('wrong TV_TYPE');
                end
            case 'IM_SIZE'
                im_size = varargin{i+1};
            case 'AL_ITERS'
                AL_iters = round(varargin{i+1});
                if (AL_iters <= 0 )
                    error('AL_iters must a positive integer');
                end
            case 'POSITIVITY'
                positivity = varargin{i+1};
                if strcmp(positivity,'yes')
                    reg_pos = 1;
                end
            case 'ADDONE'
                addone = varargin{i+1};
                if strcmp(addone,'yes')
                    reg_add = 1;
                end
            case 'MU'
                mu = varargin{i+1};
                if mu <= 0
                    error('mu must be positive');
                end
            case 'SIZEOFDICTIONARY'
                sizeOfDictionary = varargin{i+1};
                if sizeOfDictionary <= 0
                    error('size of Dictionary must be positive');
                end
            case 'WINDOW'
                w_r =  varargin{i+1};
                if w_r <= 0
                    error('size of Dictionary must be positive');
                end
            case 'LAMBDA_D'
                lamda_d =  varargin{i+1};
                 if lamda_d < 0
                    error('lambda must be non-negative');
                 end
            case 'RUNKSVD'
                runksvd  =  varargin{i+1};
                if runksvd ~= 0 || runksvd ~=1
                    error('lambda must be non-negative');
                end
            case 'DICTIONARY'
                D =  varargin{i+1};
                if D== 0 
                    error('D must be non empty');
                end  
                usePredfinedDictionary = 0;
            case 'VERBOSE'
                verbose = varargin{i+1};
            case 'X0'
                U0 = varargin{i+1};
            case 'TRUE_X'
                XT = varargin{i+1};
                true_x = 1;
            case 'SIZEX'
                sizex = varargin{i+1};
            case 'SIZEY'
                sizey = varargin{i+1};
            otherwise
                % Hmmm, something wrong with the parameter string
                error(['Unrecognized option: ''' varargin{i} '''']);
        end;
    end;
end

if ~exist('D','var')
 if ~exist('w_r','var')
     error(['Enter the window size for Dictionary: ''' varargin{i} '''']);
 end
 usePredfinedDictionary == 1;
else
    w_r = sqrt(size(D,1));
end
     

%% Default Parameters
%% Initialization
X_hat = pinv(M)*Y;

for iter = 1:2
    

    %% Step 1
    if 1==0
        X_cols = [];
            minimumExamplesFromEachImage = 1000;
            for i = 1:size(X_hat,1)
                I = reshape(X_hat(i,:),[sizex sizey]);    
                [level] = graythresh(I);
                 X=im2col(I,[w_r w_r],'sliding');
                [r c v]= find(X>level*1.75);
                X = X(:,unique(c));
                X=X-repmat(mean(X),[size(X,1) 1]);
                X=X ./ repmat(sqrt(sum(X.^2)),[size(X,1) 1]);
                if (minimumExamplesFromEachImage > size(X,2))
                    X_cols = [X_cols X];
                else
                    X_cols = [X_cols X(:,randi(size(X,2),1,minimumExamplesFromEachImage))];
                end

            end
        if runksvd == 0
            param.K=sizeOfDictionary;  % learns a dictionary with 100 elements
            param.lambda=0.85;
            param.numThreads=-1; % number of threads
            param.batchsize=400;
            param.verbose=false;

            param.iter=20;  % let us see what happens after 1000 iterations.

            %%%%%%%%%% FIRST EXPERIMENT %%%%%%%%%%%
            D = mexTrainDL(X_cols,param);
            dictimg = showdict(D,[1 1]*params.blocksize,round(sqrt(params.dictsize)),round(sqrt(params.dictsize)),'lines','highcontrast');
            figure; imshow(imresize(dictimg,2,'nearest'));
            title('Trained dictionary SPAMS');
        end

        if runksvd == 1
            %% run k-svd training %%
            params.data = X_cols;
            params.Tdata = 5;
            params.dictsize = sizeOfDictionary;
            params.iternum = 30;
            params.memusage = 'high';
            params.blocksize = w_r;
            [D,g,err] = ksvd(params,'');
            dictimg = showdict(D,[1 1]*params.blocksize,round(sqrt(params.dictsize)),round(sqrt(params.dictsize)),'lines','highcontrast');
            figure; imshow(imresize(dictimg,2,'nearest'));
            title('Trained dictionary');

        end
    end

    % reconstruct all the images from X_hat
       X_img_hat_recon_all = zeros(size(X_hat));
       parfor i = 1:size(X_hat,1)
         params = [];
         params.x= reshape(X_hat(i,:),sizex,sizey);
         params.blocksize=[w_r w_r];
         params.dict = D;
    %          params.sigma=1;
         params.psnr = 50;
         params.maxatoms= 5;
         params.stepsize=50;
         params.memusage='high';
        [x_recon,nz]= ompdenoise(params,-1);
        X_img_hat_recon_all(i,:) =  x_recon(:);
       end

    %% Step 2
    % At this point we have dictionary and reconstruction of all the images
    %         lambda = 10^-4; % [0.5*10^-4 10^-3 5*10^-3 0.01 0.05 0.1 0.3 0.5 1];
    %         % for spatial regularization
    %         lambda_TV = 0.005; %[0.5*10^-4 10^-3 5*10^-3 0.01 0.05 0.1 0.3 0.5 1];
    % %         for i=1:length(lambda_)
    % %             for j=1:length(lambda_TV_)
            Y_Appended = [sqrt(lamda_d).*Y;X_img_hat_recon_all];
            M_Appended = [sqrt(lamda_d).*M;eye(size(X_img_hat_recon_all,1))];
            [X_hat,res,rmse_ni] = sunsal_tv(M_Appended,Y_Appended,'MU',0.05,'POSITIVITY','yes','ADDONE','no', ...
                                       'LAMBDA_1',lambda_l1,'LAMBDA_TV', lambda_TV, 'TV_TYPE','niso',...
                                       'IM_SIZE',[sizex sizey],'AL_ITERS',200, 'VERBOSE',verbose);
end

end


% 
% function [data_ret] = temp_fun(data,level)
% data_ret = [];
% if(mean(data)>level)
%     data_ret = data;
% end
% 
% end
