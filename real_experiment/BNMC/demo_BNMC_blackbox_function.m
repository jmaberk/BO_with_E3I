
%fprintf('Loading Scene Data\n');
%[yyTrain, xxTrain]=read_sparse_ml('data\scene_train');
%[yyTest, xxTest]=read_sparse_ml('data\scene_test');

eta_xx=0.1; % Dirichlet symmetric for SVI (feature)
eta_yy=0.01; % Dirichlet symmetric for SVI (label)
learn_rate=0.001; % learning rate for SVI
lambda=64;% for SGD
trun_thesh=0.00001; % truncation threshold
alpha=1; % stick-breaking parameter

parameters=[eta_xx,eta_yy,learn_rate,lambda,trun_thesh,alpha];


fprintf('eta_x=%.4f eta_y=%.4f learning_rate=%.4f lambda=%.1f truncation=%.6f stick-breaking=%.2f\n',eta_xx,eta_yy,learn_rate,lambda,trun_thesh,alpha);
Y=BayesOpt_BNMC(parameters);
fprintf('y=f(X)=%.3f\n',Y);