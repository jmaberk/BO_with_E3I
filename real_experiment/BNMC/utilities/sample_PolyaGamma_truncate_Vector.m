function X = sample_PolyaGamma_truncate_Vector( aa, cc )
%SAMPLE_POLYAGAMMA Summary of this function goes here
%   a>0
%   c in R

%following Gaussian approach
%http://ml-thu.net/~scalable-ctm/scalable-ctm.pdf
pi=3.14;
KK=1000;
kk=1:KK;

NN=size(cc,1);

temp=sum(gamrnd(aa,1,[NN KK])./(ones(NN,1)*((kk-1/2).*(kk-1/2))+(cc.*cc)*ones(1,KK)./(4*pi*pi)),2);

X=temp./(2*pi*pi);
X=X';

end

