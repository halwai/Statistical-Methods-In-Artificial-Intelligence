% LMS procedure for linear classification of two classes
% input 
% X:: normalised setup of two category linearly seprable class
% theta:: intial weight vectors 
% output
% weights :: if seprable correct weights
function weights = LMS(X,theta,b)

nn= 0.05; % the ita factor

[m,d]=size(X);

limit = 10000 ; % limit in number of loops so if no convergence is found loop still exits

for s=1:limit
    
    for i=1:m
        %temp is the current data point(row vector)
        temp= X(i,:);
        % update step
        theta = theta + nn*(temp')*( b-temp*theta )/(temp*temp') ;
    end
    
end

%final_weights
weights=theta;
