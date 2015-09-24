% single sample perceptron for linear classification where error
% function is the value of the misclassifed sample
% input 
% X:: normalised setup of two category linearly seprable class
% theta:: intial weight vectors 
% output
% weights :: if seprable correct weights
function weights = single_sample_perceptron(X,theta)

nn= 0.5; % the ita factor

[m,d]=size(X);

limit = 10000 ; % limit in number of loops so if no convergence is found loop still exits

for s=1:limit
    
    flag=1;
    
    for i=1:m
        % update step 
        if X(i,:)*theta <= 0
            theta = theta + nn*X(i,:)' ;
            flag=0;
        end
    end
    
    %break if no update occours
    if flag
        break;
    end
end
%final_weights
weights=theta;
