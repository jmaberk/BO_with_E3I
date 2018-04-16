function sample = sample_inverseGaussianVector( mu, lambda )
% generate sample from inverse Gaussian distribution

%sample from a normal distribution with a mean of 0 and 1 standard deviation
N=size(mu,1);
V=size(mu,2);
v=randn(N,V);

y=v.*v;

x= mu +(mu.*mu.*y)/(2*lambda)-sqrt(4*mu.*lambda.*y+mu.*mu.*y.*y).*(mu./(2.*lambda));

test=rand(N,V);

sample=mu.*mu./x;
idx=find(test<=(mu./(mu+x)));
sample(idx)=x(idx);

% if test<=(mu./(mu+x))
%     sample =x;
%     return;
% else
%     sample=mu.*mu./x;
%     return;
% end

