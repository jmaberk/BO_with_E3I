function [ Expect_log_sticks ] = Expect_Log_Sticks( par1, par2 )
% compute the expected log of stick breaking
% E[ log \pi_k] = E [ log v_k] + \sum_j=1:k-1 E[ log (1-vj)]

mysum=psi(par1+par2);
Elogv=psi(par1)-mysum;
Elog1_v=psi(par2)-mysum;

N=length(par1)+1;
Expect_log_sticks=zeros(1,N);
%Expect_log_sticks(1)=Elogv(1);
Expect_log_sticks(1:end-1)=Elogv;

Expect_log_sticks(2:end)=Expect_log_sticks(2:end)+cumsum(Elog1_v);

end

