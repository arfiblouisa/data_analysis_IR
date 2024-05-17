function [baselined,baseline]=whittaker_baseline(y, lambda, p);
%based on "Baseline Correction with Asymmetric Least Squares
%Smoothing" by Paul H. C. Eilers Hans F.M. Boelens (2005)
m=length(y);    %calculate length of spectrum
E=speye(m);     %construct 
D=diff(E,2);    %construct difference matrix
w=ones(m,1);    %initial guess of weights
for i=1:500
W=spdiags(w,0,m,m);     %Construct weights matrix
C=chol(W+lambda*D'*D);  %
z=C\(C'\(w.*y));        %compute the estimated baseline
num_wj=0;               %reset/define parameter used to check convergence
%assign new weights for the system
for j=1:m
    if y(j)>z(j)       
        w(j)=p;
        num_wj=num_wj+1;
    elseif y(j)<z(j)
        w(j)=1-p;
    end
    num_comp(i)=num_wj;
end
% check for convergence (number of datapoints with weight w(i)=1-p)
if i>1
    if num_comp(i)-num_comp(i-1)==0
        break
    end
end
end
baseline=z; %define baseline
baselined=y-baseline; %calculate the spectrum after background subtraction
end