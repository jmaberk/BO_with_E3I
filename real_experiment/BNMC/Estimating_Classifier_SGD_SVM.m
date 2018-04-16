function [ ww ] = Estimating_Classifier_SGD_SVM( xxTrain, yyTrain )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

%% estimating Classifier

VY=size(yyTrain,2);
nTrain=size(yyTrain,1);
VX=size(xxTrain,2);

% weight for unbalance classes
weight=sum(yyTrain);
weight=weight./sum(weight);

for ii=1:nTrain
    xxTrain(ii,:)=xxTrain(ii,:).*(1-sum(weight(yyTrain(ii,:)==1)));
end


yyTrain(yyTrain==0)=-1;
lambda=32/nTrain;
nClass=VY;

loss_type=1; % Hingle loss
ww=zeros(VY,VX);
myerror=0;



for tt=1:nTrain
    %disp(tt);
    %disp(alpha)
    % select tt
    
    %idx=randperm(NN);
    %selected_idx=idx(1);
    selected_idx=tt;
    xt = xxTrain(selected_idx,:)';
    yt = yyTrain(selected_idx,:);
    
    temp=ww*xt;
    for kk=1:VY
        if temp(kk)>=0
            yypred=1;
            if yt(kk)==-1
                myerror=myerror+1;
            end
        else
            yypred=0;
            if yt(kk)==1
                myerror=myerror+1;
            end
        end
    end
    
    % evaluate mistake
    if mod(tt,500)==0
        fprintf('%d ',myerror);       
    end
    %
    %     end
    % compute first derivative of loss function
    % ywx
    % compute kernel K < xx, xx(SupportIndex) >
    %KK_xt_xB = exp(-gamma.*pdist2(xxTrain(SupportIndex,:),xx,'euclidean').^2);
    
    IsSatisfied=ones(1,VY);
    mygrad=0;
    
    switch loss_type
        case 1% Hingle
            temp=full(yt.*(ww*xt)');% hingle loss
            IsSatisfied(temp>=1)=0;
            mygrad=-xt*yt;
        case 2% L2
            temp=full(ww'*xt);
            if temp==yt
                IsSatisfied=0;
            end
            mygrad=(temp-yt)*xt;
        case 3 % L1
            temp=full(ww'*xt);
            if temp==yt
                IsSatisfied=0;
            end
            mygrad=sign(temp-yt)*xt;
            
        case 4 % Logistic
            temp=full(-yt*ww'*xt);
            mygrad=-xt*yt*exp(temp)/(exp(temp)+1);
        case 5 % Epsilon
            temp=full(ww'*xt)-yt;
            if (abs(temp)<=epsilon) && (temp==0)
                IsSatisfied=0;
            end
            mygrad=sign(temp)*xt;
    end
    
    %if IsSatisfied==1 % first derivative not zero
    %eta=1/(lambda*tt);
    %ww=ww-eta*mygrad;
    %end
    eta=1/(lambda*tt);
    mygrad=mygrad';
    ww(IsSatisfied==1,:)=ww(IsSatisfied==1,:)-eta*mygrad(IsSatisfied==1,:);
    
end

ww=ww';
end

