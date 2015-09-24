%%NEURAL NET BATCH BACK -PROPAGATION WITH 3 LAYERS%%
% activation fucntion is assumed  sigmoid 
% INPUT
% x :: original data
% t :: expected output of data(target parameter result of preprocessing)
% nn :: ita
% mid :: number of units in layer2
% fin :: number of units in layer 3
% OUTPUT
% w :: weights from input to middle layer
% s :: weigths from middle to final layer

function [w,s] = neural_net_back_propagation(x,t,nn,mid,fin)

[m,d]=size(x);
w=zeros(m,mid);
s=zeros(m,fin);

limit=1000;
for r=1:limit % limited number of epochs
    % calculate the values of y and z wrt current w,s;
    % values of y,z to be used later in computation in this epoch
    y=zeros(m,mid);
    z=zeros(m,fin);
    for i=1:d   
        temp=repmat(w(i,:)',[1 m])*x(:,i)';
        y = y + temp'; 
    end
    for i=1:mid
        temp=repmat(s(i,:)',[1 m])*x(:,i)';
        z = z + temp'; 
    end
    
    % calculate the update values for w and s
    for i=1:m % loop on number of samples
  %      for k=1:mid % loop on number of middle layers
   %         for j=1:d % loop on number of dimensions
                temp_w = temp_w + nn*x;  
    end
    
    % dont forget bias
    for i=1:m % loop on number of samples
        temp_s = temp_s + nn*(t(i,:)-z(i,:))*z(i,:)*(1-z(i,:))*y(i,:);  
    end

    %update w and s
    w = w + temp_w;
    s = s + temp_s;
    
    % break if max update val is less than a threshold 
    if max( max(abs(temp_s(:))) , max(abs(temp_w(:)) ) ) <  0.01 
        break;
    end
    
end

end
