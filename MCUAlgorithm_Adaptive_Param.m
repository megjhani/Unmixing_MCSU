function [ S_hat A ] = MCUAlgorithm_Adaptive_Param( Y,r,T,noOfSources,sigma,lamda,M,D_s,A,size_x,size_y )
%MCUALGORITHM Summary of this function goes here
%   Observation matrix Y
%   r patch size;k dictonary size,n number of sources, lamda regularization
%   parameter, M total number of iterations


%% initialization
displayFigure = 0;
lamdavals = [];

A_temp1=zeros(size(A));
S_hat = (A')*Y;

% 
%     for i = 1:size(S_hat,1)+1
%         eval(['fileName_',num2str(i),' =''','file_',num2str(i),'.gif'';']);
%     end

% for s_fig = 1:size(S_hat ,1)
%     figure(3+s_fig); imshow(reshape(S_hat(s_fig,:),size_x,size_y)/max(S_hat(s_fig,:)),[]);
%     F = getframe(3+s_fig);
%     aviobj = addframe(aviobj,F);
%     eval(['aviobj',num2str(s_fig),' = addframe(','aviobj',num2str(s_fig),',F);']);
% end

param.L=15; % not more than 5 non-zeros coefficients
param.eps=0.1; % squared norm of the residual should be less than 0.1
param.numThreads=-1; % number of processors/cores to use; the default choice is -1
 R=zeros(r*r,size_x*size_y);
strtidx = 1;
endidx = 1;
% for z = 1:r
%     R(strtidx:strtidx+r-1,endidx:endidx+r-1)=eye(r);
%     strtidx = strtidx+r;
%     endidx = endidx+size_x;
% end
% R=sparse(R);

%% begin
for i = 1:M
    
%         eval(['fig_',num2str(2),' = figure(',num2str(2),');']);
        params.lambda=adaptiveLamda(A,10,0.01);
        lamdavals = [lamdavals  params.lambda];
       if displayFigure
            fig_2 = figure(2);set(gcf,'Color', 'White','units','normalized','outerposition',[0 0 1 1]);
            plot(A,'LineWidth',4); set(gca,'FontSize',15)
            xlabel('Spectral Band','FontSize',20)
            ylabel('Trasnmittance (AU)','FontSize',20)
            title(strcat('Iteration : ',num2str(i)),'FontSize',20);
            tightfig;
            for s_fig = 1:size(S_hat ,1)
                eval(['fig_',num2str(2+s_fig),' = figure(',num2str(2+s_fig),');']);
    %                  figure;imshow(reshape(S_hat(i,:),size(s_k))/max(S_hat(i,:)));
                imshow(reshape(S_hat(s_fig,:),size_x,size_y)/max(S_hat(s_fig,:))); title(strcat('Iteration : ',num2str(i),'Source : ',num2str(s_fig)),'FontSize',20);
                set(gcf,'Color', 'White','units','normalized','outerposition',[0 0 1 1]);
                tightfig;
            end
            %% save figure(2)
            for s_fig = 1:size(S_hat,1)+1
                frame = getframe(eval(['fig_',num2str(1+s_fig)]));
                im = frame2im(frame);
                [imind,cm] = rgb2ind(im,256);
                filenametemp = ['fileName_',num2str(s_fig),'.gif'];
                if i == 1;
                    imwrite(imind,cm,filenametemp,'gif', 'Loopcount',inf);
                else
                  imwrite(imind,cm,filenametemp,'gif','WriteMode','append');
                end
            end
            
            %%% save adaptive lambda values
            fig_for_lamda = figure(s_fig+2);set(gcf,'Color', 'White','units','normalized','outerposition',[0 0 1 1]);
            plot(lamdavals,'LineWidth',4); set(gca,'FontSize',15)
            xlabel('Iteration','FontSize',20)
            ylabel('Lambda Values(AU)','FontSize',20)
            title(strcat('Iteration : ',num2str(i)),'FontSize',20);
            tightfig;
            
            frame = getframe(fig_for_lamda);
            im = frame2im(frame);
            [imind,cm] = rgb2ind(im,256);
            filenametemp = ['fileName_',num2str(s_fig+2),'.gif'];
             if i == 1;
                    imwrite(imind,cm,filenametemp,'gif', 'Loopcount',inf);
             else
                  imwrite(imind,cm,filenametemp,'gif','WriteMode','append');
             end
        end
      
%         if 1==1
% %             avifile(filename, ParameterName, ParameterValue)
%             F = getframe(fig_2);
%             aviobj = addframe(aviobj,F);
% 
%         end

    
     
    
    
    
    
    
    
%     
%     if i ~= 1
%         if sum(sum((A_pre-A).*(A_pre-A))) < 1e-4
%             break;
%         end
%         A_pre = A;
%     else
%         A_pre = A;
%     end

    for j = 1:noOfSources
            E_dc = Y-A*S_hat;
            
%         s= reshape(S_hat(j,:),[size_x size_y]);
%         s_cols = im2col(s,[r r],'sliding');
        %%% solve OMP problem
%         alpha=mexOMP(s_cols ,D_s{j},param);
        tempsum =zeros(size(A(:,j)*S_hat(j,:)));
        for sa = 1:size(A,2)
            if(sa~=j)
                tempsum = tempsum+A(:,sa)*S_hat(sa,:);
            end
            
        end
        E_j =   Y-tempsum;
%         close all;
%        for tempidx = 1:size(Y,1)
%          figure;imshow(reshape(E_j(tempidx,:),size_x,size_y)/max(E_j(tempidx,:)));
%        end
        %% compute S_hat(j,:)
        %%% Required:
        %     'x'                   signal to denoise
        %     'blocksize'           size of block to process
        %     'dict'                dictionary to denoise each block
        %     'psnr' / 'sigma'      noise power in dB / standard deviation
        %
        %   Optional (default values in parentheses):
        %     'stepsize'            distance between neighboring blocks (1)
        %     'maxval'              maximal intensity value (1)
        %     'memusage'            'low, 'normal' or 'high' ('normal')
        %     'noisemode'           'psnr' or 'sigma' ('sigma')
        %     'gain'                noise gain (1.15)
        %     'lambda'              weight of input signal (0.1*maxval/sigma)
        %     'maxatoms'            max # of atoms per block (prod(blocksize)/2)
        
        params.x=reshape((E_j)'*A(:,j),size_x,size_y);
        
        
        
%         blocksize = 2000;
%         for dc_rm = 1:blocksize:size(params.x,2)
%           blockids = dc_rm : min(dc_rm+blocksize-1,size(params.x,2));
%           params.x(:,blockids) = remove_dc(params.x(:,blockids),'columns');
%         end

        
        params.blocksize=[r r];
        params.dict = D_s{j};
%         params.sigma=sigma;
     params.psnr = 50;

%         params.maxatoms=5;

        if displayFigure
            figure(10);plot(lamdavals);
        end
        
         params.maxatoms= T;
         params.stepsize=10;
         params.memusage='high';
       [S_img_hat,nz]= ompdenoise(params,-1);
        
        S_hat(j,:)=S_img_hat(:);%/max(S_img_hat(:));
%         
        tempsum =zeros(size(A(:,j)*S_hat(j,:)));
        for sa = 1:size(A,2)
            if(sa~=j)
                tempsum = tempsum+A(:,sa)*S_hat(sa,:);
            end
            
        end
        E_j =   Y-tempsum;
%%      update A
%         E_j(find(E_j>0 & E_j <1e-3))=0;
%         E_j(find(E_j<0 & E_j >-1e-3))=0;
%         E_j  = Y-
        A(:,j) = (E_j)*(S_hat(j,:)');
        if min( A(:,j))<0
            A(:,j) = A(:,j)+-1*min(A(:,j));
        end
        A(:,j) = A(:,j)/norm(A(:,j));
        A(:,j) = normc(A(:,j));
%             if min(A(:)) < 0
%                 A = A+-1*min(A(:));
%             end
          A = normc(A);
%         A = (A-min(A(:))) ./ (max(A(:)-min(A(:))));

        if 1==0
           [S_temp A_temp] = SparseMixingMatrixRecovery(E_j,size(A,2)); 
           A_temp1(:,j)=A_temp(:,j);
           figure(3);plot(A_temp1);
           
        end
%         fig_2 = figure(2);plot(A); set(gca,'FontSize',15)
%         xlabel('Wavelength \lambda (nm)','FontSize',20)
%         ylabel('Trasnmittance (AU)','FontSize',20)
%         title(strcat('Iteration : ',num2str(i)),'FontSize',20);
%         %% save figure(2)
%         if 1==1
% %             avifile(filename, ParameterName, ParameterValue)
%             F = getframe(fig_2);
%             aviobj = addframe(aviobj,F);
% 
%         end
        
        
%%
%         close all
%         for tempidx=1:size(S_hat,1)
%            figure;imshow(params.x/max(params.x(:)))
%             figure;imshow(reshape(S_hat(tempidx,:),size_x,size_y)/max(S_hat(tempidx,:)));
%         end
        
        
        
        
        
%         
%         tempsum=zeros(size(R'*R));
%         tempR = R;
%         for i=1:size(R,2)
%             tempsum = tempsum+tempR'*tempR;
%             tempR = circshift(tempR,1,2);
%         end
        
        
        
% % % % % % % % % % % % % % % % % % % %         R(1:r,1:r)=eye(r);
% % % % % % % % % % % % % % % % % % %         temp1=[];
% % % % % % % % % % % % % % % % % % %         tic
% % % % % % % % % % % % % % % % % % %         [m, n] = size(R);
% % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % %         tempR = R;
% % % % % % % % % % % % % % % % % % %         P=1:size(R,2);
% % % % % % % % % % % % % % % % % % %         for t = 1:size(R,2)-r*r-1
% % % % % % % % % % % % % % % % % % % %             if (mod(t,size_x)+r-1>size_x || mod(t,size_x)==0) 
% % % % % % % % % % % % % % % % % % % %                 tempR = circshift(R,t,2);
% % % % % % % % % % % % % % % % % % % %                 continue;
% % % % % % % % % % % % % % % % % % % %             else
% % % % % % % % % % % % % % % % % % % % %             temp1=[temp1 tempR*S_hat(j,:)'];
% % % % % % % % % % % % % % % % % % % %             tempR = circshift(tempR,1,2);
% % % % % % % % % % % % % % % % % % % %             end
% % % % % % % % % % % % % % % % % % %             
% % % % % % % % % % % % % % % % % % % %           tempR = circshift(R,t,2);
% % % % % % % % % % % % % % % % % % % %           tempR=[R(:,n-1+1:n) R(:,1:n-1)];
% % % % % % % % % % % % % % % % % % %             R(:,t);
% % % % % % % % % % % % % % % % % % % %             circshift(P,t,2);
% % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % %         end
%         tend=toc
%         fprintf('%f signals processed per second\n',tend);

    end
    
    
    
end
%     close(fig);
%     aviobj = close(aviobj);
%     aviobj1 = close(aviobj1);
% 	aviobj2 = close(aviobj2);
%     aviobj3 = close(aviobj3);
        if displayFigure
            for s_fig = 1:size(S_hat,1)+1
                        frame = getframe(eval(['fig_',num2str(1+s_fig)]));
                        im = frame2im(frame);
                        [imind,cm] = rgb2ind(im,256);
                        filenametemp = ['fileName_',num2str(s_fig),'.gif'];
                        if i == 1;
                            imwrite(imind,cm,filenametemp,'gif', 'Loopcount',inf);
                        else
                          imwrite(imind,cm,filenametemp,'gif','WriteMode','append');
                        end
            end
        end
end

function [adaptive_lamda] = adaptiveLamda(A,alpha,beta)
    alpha = 1;% controls the magintude of the exponential curve
    beta = 0.01; % controls the smoothness of the exponential curve
    smallest_angle = 0;
    angle_vector = [];
    for i = 1:size(A,2)
        for j = i+1:size(A,2)
            u = A(:,i);
            v = A(:,j);
            cosTheta = dot(u,v)/(norm(u)*norm(v));
            thetaInDegrees = acos(cosTheta)*180/pi;
            angle_vector = [angle_vector; thetaInDegrees];
        end
    end
    smallest_angle = min(angle_vector);
    adaptive_lamda = alpha*exp(-(smallest_angle.*beta));
end

