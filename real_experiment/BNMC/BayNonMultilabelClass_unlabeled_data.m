function [pred_yyTest] = BayNonMultilabelClass_unlabeled_data( xxTrain,yyTrain,xxTest, options )
%  input ===================================================================
%   xxTrain: feature of training data [NTrain x DD]
%   yyTrain: label of trainning data [NTrain x CC]
%   xxTest: feature of testing data [NTest x DD]
%   options=11; % Batch Setting, SVM
%   options=12; % Batch Setting, LR
%   options=21; % Online Setting, SVM
%   options=22; % Online Setting, LR
%   =========================================================================
%   output ==================================================================
%   pred_yyTest = predicted label : [NTest x CC]
%   contact: anonymous

% learning rate
eta_xx=0.1;
eta_yy=0.01;
learn_rate=0.001; % rho in the paper

KK=20;
VX=size(xxTrain,2);
VY=size(yyTrain,2);

nTrain=size(xxTrain,1);
nTest=size(xxTest,1);

mean_number_yy=mean(sum(yyTrain,2));
std_number_yy=std(sum(yyTrain,2));

wwTrain=mat2cell(xxTrain,ones(1,nTrain),VX);
wwTest=mat2cell(xxTest,ones(1,nTest),VX);
llTrain=mat2cell(yyTrain,ones(1,nTrain),VY);

% customize for SGD estimation
yyTrain_sgd=yyTrain;
yyTrain_sgd(yyTrain==0)=-1;

% scale data to prevent bias estimation in SGD
weight=sum(yyTrain);
weight=weight./sum(weight);

xxTrain_sgd=xxTrain;
for ii=1:nTrain
    xxTrain_sgd(ii,:)=xxTrain(ii,:).*(1-sum(weight(yyTrain(ii,:)==1)));
end

% lambda in SGD
lambda=64/nTrain;

phi_xx=rand(KK,VX);
psi_yy=rand(KK,VY);

alpha=1;
pi_1=ones(1,KK);
pi_2=alpha*ones(1,KK);

eta_classifier=zeros(VY,VX);
ww_array=cell(1,nTrain);

for tt=1:2 % repeat twice for better estimation
    for nn=1:nTrain
        if mod(nn,5000)==0
            fprintf('%d ',nn);
        end
        
        Sigma_V=Expect_Log_Sticks(pi_1,pi_2);
        
        exp_log_phi=Dirichlet_Expectation(phi_xx);
        exp_log_psi=Dirichlet_Expectation(psi_yy);
        
        % estimate zz  - step 2 in Algorithm 2.
        
        if sum(yyTrain(nn,:))==0
            zz=Sigma_V(1:KK)'+exp_log_phi*full(wwTrain{nn})'./sum(wwTrain{nn});
        else
            zz=Sigma_V(1:KK)'+exp_log_psi*full(llTrain{nn})'./sum(llTrain{nn})+...
                exp_log_phi*full(wwTrain{nn})'./sum(wwTrain{nn});
        end
        
        
        zz=exp(zz-max(zz));
        zz=zz./sum(zz);
        
        global_zz(nn,:)=zz;
        
        nat_grad_phi_xx=zz*full(wwTrain{nn});
        nat_grad_psi_yy=zz*full(llTrain{nn});
        
        nat_grad_pi_1=1+nTrain*zz;
        
        temp=cumsum(zz);
        nat_grad_pi_2=alpha+2*nTrain*(1-temp);
        
        if nn==1
            init_learn_rate=0.99;
            % step 3 in Algorithm 2.
            phi_xx=(1-init_learn_rate)*phi_xx+init_learn_rate*(eta_xx+nTrain*nat_grad_phi_xx);
            % step 4 in Algorithm 2.
            if sum(yyTrain(nn,:))~=0
                psi_yy=(1-init_learn_rate)*psi_yy+init_learn_rate*(eta_yy+nTrain*nat_grad_psi_yy);
            end
            
            % step 5 in Algorithm 2.
            pi_1=(1-init_learn_rate)*pi_1+init_learn_rate*nat_grad_pi_1';
            pi_2=(1-init_learn_rate)*pi_2+init_learn_rate*nat_grad_pi_2';
        else
            % step 3 in Algorithm 2.
            phi_xx=(1-learn_rate)*phi_xx+learn_rate*(eta_xx+2*nTrain*nat_grad_phi_xx);
            % step 4 in Algorithm 2.
            if sum(yyTrain(nn,:))~=0
                psi_yy=(1-learn_rate)*psi_yy+learn_rate*(eta_yy+2*nTrain*nat_grad_psi_yy);
            end
            
            % step 5 in Algorithm 2.
            pi_1=(1-learn_rate)*pi_1+learn_rate*nat_grad_pi_1';
            pi_2=(1-learn_rate)*pi_2+learn_rate*nat_grad_pi_2';
        end
        
        % ignore if missing label
        if sum(yyTrain(nn,:))~=0
            % step 6 in Algorithm 2.
            switch options
                case 21
                    xt = xxTrain_sgd(nn,:)';
                    yt = yyTrain_sgd(nn,:);
                    IsSatisfied=ones(1,VY);
                    temp=full(yt.*(eta_classifier*xt)');% hingle loss
                    IsSatisfied(temp>=1)=0;
                    mygrad=-xt*yt;
                    eta=1/(lambda*nn);% learning rate
                    mygrad=mygrad';
                    eta_classifier(IsSatisfied==1,:)=eta_classifier(IsSatisfied==1,:)-eta*mygrad(IsSatisfied==1,:);
                    ww_array{nn}=eta_classifier;
                    
                case 22
                    xt = xxTrain_sgd(nn,:)';
                    yt = yyTrain_sgd(nn,:);
                    
                    temp=full(-yt.*(eta_classifier*xt)');% logistic loss
                    
                    temp(temp>100)=100;% smoothing
                    mygrad=-xt*(yt.*exp(temp)./(exp(temp)+1));
                    
                    eta=1/(lambda*nn);% learning rate
                    mygrad=mygrad';
                    
                    eta_classifier=((nn-1)/nn)*eta_classifier-eta*mygrad;
                    ww_array{nn}=eta_classifier;
            end
        end
        
    end
end

% truncation step in SVI to remove the empty topic
temp_sum=sum(global_zz);
idx=find(temp_sum<0.00001*nTrain);
phi_xx(idx,:)=[];
psi_yy(idx,:)=[];

KK=KK-length(idx);

% normalize phi and psi
phi_xx=bsxfun(@rdivide,phi_xx,sum(phi_xx,2));
psi_yy=bsxfun(@rdivide,psi_yy,sum(psi_yy,2));

fprintf('KK=%d\n',KK);

%% plot phi and psi
figure;
imagesc(phi_xx);
set(gca,'fontsize',12);
ylabel('Correlations','fontsize',14);
xlabel('Features','fontsize',14);
%set(gca,'YTick',[1 2 3 4 5])
%set(gca,'YTickLabel',[1 2 3 4 5]);
title('\psi_k','Interpreter','Tex');

figure;
imagesc(psi_yy);
set(gca,'fontsize',12);
xlabel('Classes','fontsize',14);
ylabel('Correlations','fontsize',14);
%set(gca,'YTick',[1 2 3 4 5])
%set(gca,'YTickLabel',[1 2 3 4 5]);
title('\phi_k','Interpreter','Tex');

%% estimating eta

temp=sum(yyTrain,2);
idxZero=find(temp==0);
yyTrain(idxZero,:)=[];
xxTrain(idxZero,:)=[];

switch options
    case 11
        fprintf('BNMC-SVM')
        [ eta_classifier] = Estimating_Classifier_SVM( xxTrain, yyTrain );
    case 12
        fprintf('BNMC-LR')
        [ eta_classifier] = Estimating_Classifier_Logistic( xxTrain, yyTrain );
    case 21
        fprintf('BNMC-Online-SVM')
        %eta_classifier will automatically be taken from previous step
        eta_classifier=eta_classifier';
    case 22
        fprintf('BNMC-Online-LR')
        eta_classifier=eta_classifier';
        %eta_classifier will automatically be taken from previous step
end



%% test set
global_zzTest=zeros(nTest,KK);

exp_log_phi=Dirichlet_Expectation(phi_xx);

for nn=1:nTest
    Sigma_V=Expect_Log_Sticks(pi_1,pi_2);
    
    phi=Sigma_V(1:KK)'+exp_log_phi*full(wwTest{nn})'./sum(wwTest{nn});
    
    phi=exp(phi-max(phi));
    phi=phi./sum(phi);
    
    global_zzTest(nn,:)=phi;
end

prob_prior_yyTest=global_zzTest*psi_yy;

prob_classifier_yyTest=SigmoidFunction(xxTest*eta_classifier);

prob_predicted_yyTest=prob_classifier_yyTest.*prob_prior_yyTest;
prob_predicted_yyTest=bsxfun(@rdivide,prob_predicted_yyTest,sum(prob_predicted_yyTest,2));

pred_yyTest=zeros(nTest,VY);

% generate T
TT = normrnd(mean_number_yy,std_number_yy^2,nTest,1);
TT=ceil(TT);
for ii=1:nTest
    for tt=1:TT(ii)
        uu = rand;
        kk = 1+sum(uu>cumsum(prob_predicted_yyTest(ii,:)));
        pred_yyTest(ii,kk)=1;
    end
end

end

