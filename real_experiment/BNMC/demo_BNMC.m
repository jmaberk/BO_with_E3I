% Bayesian Nonparametric Multilabel Classification demo
clear all;
close all;
addpath(genpath(pwd))

%% loading data

fprintf('Loading Scene Data\n');
%[yyTrain, xxTrain]=read_sparse_ml('data\scene_train');
%[yyTest, xxTest]=read_sparse_ml('data\scene_test');

load('SceneData.mat');

if isempty(xxTrain) || isempty(xxTest)
    return;
end

% convert from sparse to full matrix for fast matrix multiplication
yyTest=full(yyTest);
xxTrain=full(xxTrain);
xxTest=full(xxTest);
yyTrain=full(yyTrain);

%% plot the input data
figure;
imagesc(xxTrain);
set(gca,'fontsize',12);
ylabel('Data Points','fontsize',14);
xlabel('Features','fontsize',14);
title('Ground Truth Feature','fontsize',14);

figure;
imagesc(yyTrain);
set(gca,'fontsize',12);
ylabel('Data Points','fontsize',14);
xlabel('Classes','fontsize',14);
title('Ground Truth Label','fontsize',14);

close all;
%%
start=tic;
options=11; % Batch Setting, SVM
%options=12; % Batch Setting, LR
%options=21; % Online Setting, SVM
%options=22; % Online Setting, LR


IsPlot=0; % IsPlot=0 if you dont want to plot

[pred_yyTest] = BayNonMultilabelClass( xxTrain,yyTrain,xxTest, options, IsPlot );
mytime=toc(start);

%% Evaluation
[ Scores ] = MultiLabelEvaluation( yyTest,pred_yyTest );
F1Test=Scores.F1;
ExactScoreTest=Scores.ExactMatch;

fprintf(' F1=%.3f Exact=%.4f Running Time=%.1f\n',F1Test,ExactScoreTest,mytime);


