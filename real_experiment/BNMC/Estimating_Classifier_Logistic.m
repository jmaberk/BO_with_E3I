function [ eta_classifier ] = Estimating_Classifier_Logistic( xxTrain, yyTrain )
%   Estimating Logistic  Regression parameter following Polson 2013 Augmentation Approach.
%   input ===================================================================
%   xxTrain: feature of training data [NTrain x DD]
%   yyTrain: label of trainning data [NTrain x CC]
%   output ==================================================================
%   eta_classifier: Logistic Regression classifier [CC x DD]

%% estimating Classifier Logistic Regression eta_c in batch setting

VY=size(yyTrain,2);
nTrain=size(yyTrain,1);
VX=size(xxTrain,2);

nClass=VY;

lambda=0.01*ones(nTrain,VY);
eta_classifier=ones(VX,nClass);
mu_post2=zeros(nTrain,VY);


for iter=1:5   
    % estimating auxiliaray variable lambda
    for cc=1:VY
        idx=find(yyTrain(:,cc)==1);
        mu_post2(idx,cc)=xxTrain(idx,:)*eta_classifier(:,cc);
        lambda(idx,cc)=sample_PolyaGamma_truncate_Vector(1,mu_post2(idx,cc));
    end
    
    % estimating eta
    for cc=1:nClass       
        tu = bsxfun(@times, xxTrain, lambda(:,cc));
        inv_sigma_post=eye(VX)+tu'*xxTrain;
        
        temp_yy=-1*ones(nTrain,1);
        temp_yy(yyTrain(:,cc)==1)=1;
        sumtempmu=xxTrain'*temp_yy;
        mu_post=inv_sigma_post\(sumtempmu  );
        eta_classifier(:,cc)=mu_post;
    end
end


end

