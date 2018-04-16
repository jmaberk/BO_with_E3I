function [ eta_classifier ] = Estimating_Classifier_SVM( xxTrain, yyTrain )
%   Estimating SVM parameter following Polson 2011 Augmentation Approach.
%   input ===================================================================
%   xxTrain: feature of training data [NTrain x DD]
%   yyTrain: label of trainning data [NTrain x CC]
%   output ==================================================================
%   eta_classifier: svm classifier [CC x DD]

%% estimating Classifier SVM eta_c in batch setting

VY=size(yyTrain,2);
nTrain=size(yyTrain,1);
VX=size(xxTrain,2);

yyTrain_PosNeg=yyTrain; % for multiplication in computing classifier 
yyTrain_PosNeg(yyTrain_PosNeg==0)=-1;


nClass=VY;

eta_classifier=ones(VX,nClass);
mu_post2=zeros(nTrain,VY);


for iter=1:5
    
    % estimating auxiliaray variable lambda
    for cc=1:VY
        mu_post2(:,cc)=1./(abs(1-yyTrain_PosNeg(:,cc).*(xxTrain*eta_classifier(:,cc))));
    end

    mu_post2(mu_post2<0.000001)=0.000001;
    mu_post2(isinf(mu_post2))=100000;

    inv_lambda=sample_inverseGaussianVector(mu_post2,1);
    
    % estimating eta
    for cc=1:nClass
        tu = bsxfun(@times, xxTrain, inv_lambda(:,cc));
        inv_sigma_post=eye(VX)+tu'*xxTrain;
        
        temp=(ones(nTrain,1)+inv_lambda(:,cc))*ones(1,VX);
        temp_jj=xxTrain.*temp;
        
        temp_yy=-1*ones(nTrain,1);
        temp_yy(yyTrain(:,cc)==1)=1;
        sumtempmu=temp_jj'*temp_yy;
        mu_post=inv_sigma_post\(sumtempmu  );
        eta_classifier(:,cc)=mu_post;
    end
end

end

