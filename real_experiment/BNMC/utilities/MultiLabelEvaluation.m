function [ Scores ] = MultiLabelEvaluation( yyGroundTruth,yyPredicted )
% Evaluation Multilabel Classification
%  input ===================================================================
%   yyGroundTruth: true label of data [NTrain x CC]
%   yyPredicted: label of predicted values [NTrain x CC]
%   =========================================================================
%   output ==================================================================
%   Scores.F1 = F1 score
%   Scores.ExactMatch = Exact Match score
%   contact: anonymous


NN=size(yyGroundTruth,1);
NL=size(yyGroundTruth,2);
% F1
% Micro %Across Datapoints
Precision=zeros(1,NN);
Recall=zeros(1,NN);
F1_NN=zeros(1,NN);
TP=0;
TN=0;
FP=0;
FN=0;

idxPos=find(yyGroundTruth==1);
idxNeg=find(yyGroundTruth==0);
Pos=length(idxPos);
Neg=NN*NL-Pos;

TP=sum(yyGroundTruth(idxPos)==yyPredicted(idxPos));
TN=sum(yyGroundTruth(idxNeg)==yyPredicted(idxNeg));
FP=Neg-TN;
FN=Pos-TP;

Precision=TP/(TP+FP);
Recall=TP/(Pos);
F1=2*Precision*Recall/(Precision+Recall);

% exact match
temp_minus=abs(yyPredicted-yyGroundTruth);
temp_sum=sum(temp_minus,2);
ExactMatchScore=length(find(temp_sum==0))/NN;


Scores.F1=F1;
Scores.ExactMatch=ExactMatchScore;
end

