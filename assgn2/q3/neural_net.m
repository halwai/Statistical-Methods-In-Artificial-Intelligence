tic
training = csvread('train.txt');

class = training(:, size(training, 2));
output = zeros(size(training, 1), 2);
for i=1:size(class, 1)
    if class(i) == 7
        output(i, :) = [0 1];
    else
        output(i, :) = [1 0];
    end
end
training = training(:, 1:size(training, 2)-1);
training = [ones(size(training, 1), 1) training];

%size(training)
features = 65;
hidden_units = 70;
output_units = 2;

wij = rand(features, hidden_units)*2/(features^0.5) - 1/(features^0.5);
%wij = reshape(wij, [features, hidden_units]);
wjk = rand(hidden_units, output_units)*2/(hidden_units^0.5) - 1/(hidden_units^0.5);
%wjk = reshape(wjk, [hidden_units, output_units]);
eta = 1;

theta = 0.001;
m = 1;

iter = 0;
while 1
    sample = training(m, :);
    
    netj = sample*wij;
    y = (logsig(netj));
    netk = y*wjk;
    z = (logsig(netk));
    
    J = 0.5*(norm(output(m, :)-z)^2);
    
    delk = (output(m, :)-z).*(logsig(netk).*(1 - logsig(netk)));
    
    delj = delk*wjk'.*(logsig(netj).*(1 - logsig(netj)));
    
    %wij = wij + eta*(sample'*delj);
    %wjk = wjk + eta*(y'*delk);  
    wjk = wjk + eta*repmat(delk, hidden_units, 1).*[y; y]';
    wij = wij + eta*repmat(sample', 1, hidden_units).*repmat(delj, features, 1);
    
    if abs(output(m, :)-z) < theta
       
    end
    iter = iter+1;
    if(m == size(training, 1))
       break; 
    end
    m = mod(m, size(training, 1))+1;
    
    %m = mod(m, size(training, 2))+1;
end

test = csvread('test.txt');

class = test(:, size(test, 2));
test = test(:, 1:size(test, 2)-1);
test = [ones(size(test, 1), 1) test];

count = 0;
correct = 0;
c = 0;
m = 1;
for i=1:size(test, 1)
    count = count+1;
    sample = test(i, :);
    
    netj = sample*wij;
    y = (logsig(netj));
    netk = y*wjk;
    z = round(logsig(netk));
    
    if z(1) == 0 && class(i) == 7
        correct = correct + 1;
    elseif z(1) == 1 && class(i) == 0
        correct = correct + 1;
    end
end

accuracy = 100*correct/count;
accuracy
toc