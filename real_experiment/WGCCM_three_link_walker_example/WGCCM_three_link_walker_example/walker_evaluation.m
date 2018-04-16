function [ hz_velocity ] = walker_evaluation( a )
%WALKER_EVALUATION Summary of this function goes here
%   Detailed explanation goes here

global t_2 torque y force

torque = [];
t_2 = [];
y = [];
force = [];

tstart = 0;
tfinal = 13;

%% The optimization parameters
%
%a = [0.512 0.073 0.035 -0.819 -2.27 3.26 3.11 1.89];
%a = [0.512 0.053 0.035 -0.319 -6.27 -3.26 -3.11 -1.89];
%a=rand(1,8);


omega_1 = 1.55;
x0 = sigma_three_link(omega_1,a);
x0 = transition_three_link(x0).';
x0 = x0(1:6);

options = odeset('Events','on','Refine',4,'RelTol',10^-5,'AbsTol',10^-6);

tout = tstart;
xout = x0.';
teout = []; xeout = []; ieout = [];

%disp('(impact ratio is the ratio of tangential to normal');
%disp('forces of the tip of the swing leg at impact)');

for i = 1:5 % run five steps
    % Solve until the first terminal event.
    [t,x,te,xe,ie] = ode45('walker_main',[tstart tfinal],x0,options,a);
    
    if isempty(ie)==1
        hz_velocity=0;
        return;
    end
    
    % Accumulate output.  tout and xout are passed out as output arguments
    nt = length(t);
    tout = [tout; t(2:nt)];
    xout = [xout;x(2:nt,:)];
    teout = [teout; te]; % Events at tstart are never reported.
    xeout = [xeout; xe];
    ieout = [ieout; ie];
    
    % Set the new initial conditions (after impact).
    x0=transition_three_link(x(nt,:));
    
    % display some useful information to the user
    disp(['step: ',num2str(i),', impact ratio:  ',num2str(x0(7)/x0(8))])
    
    % Only positions and velocities needed as initial conditions
    x0=x0(1:6);
    
    tstart = t(nt);
    if tstart>=tfinal
        break
    end
end

%disp('by Eric R. Westervelt, Jessy W. Grizzle,');
%disp('Christine Chevallereau, Jun-Ho Choi, and Benjamin Morris');

hz_velocity=anim(tout,xout,1/30,1);


end



function hz_velocity=anim(t,x,ts,speed)

[n,m]=size(x);
[vV,vH]=hip_vel(x); % convert angles to horizontal position of hips

%hz_velocity=max(vH);
%hz_velocity=vH(end);
hz_velocity=mean(vH);

end



%% --------------------------------------------------------------------------
%% a function to calculate hip velocity

function [vV,vH] = hip_vel(x)

vV=zeros(length(x),1);
vH=cos(x(:,1)).*x(:,4); % estimate of horizontal velocity of hips

end

