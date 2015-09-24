%%  p fold k-nn classification %%

%input %
%caution -- all inputs shold be of type double do the necessary pre-processing
%data - n*d matrix : n sample points , d dimensions of a sample point
%gt - n*1 mattrix : ground truth for each corresponding data point
% p_max - 2-p folds
% k_max - k-nearesrt

%output%

function [accuracy,deviation]=myknn(data,gt,k_max,p_max)

[n,d]=size(data);
accuracy=zeros(p_max,k_max);
deviation=zeros(p_max,k_max);

unique_gt=unique(gt);

%partition set via kfold - test and training  ::  data and gt
for i=2:p_max
    c=cvpartition(n,'KFold',i);
    observed_result=zeros(k_max,c.NumTestSets);
    for j=1:c.NumTestSets
        training_data = data ( find ( c.training(j) ) , : ) ;
        test_data = data ( find ( c.test(j) ) , : ) ;
        gt_training_data = gt ( find ( c.training(j) ) , : );
        gt_testdata = gt ( find ( c.test(j) ) ,: );
        
        % k_max minimum distances of each test point from all the training points
        %different distance functions tried here
        %city-block,mahalanobis,euclidean ,seuclidean
        [distance,index]=pdist2(training_data,test_data,'mahalanobis','smallest',k_max);
        index=index.';
        distance=distance.';
        prediction=gt_training_data(index);

        % loop in k 
        for k=1:k_max
            temp_prediction=prediction(:,1:k);
            temp_distance=distance(:,1:k);
            %normalizing the distances in each row
            temp_distance=temp_distance./repmat( sum(temp_distance,2),1,k );
            
            predict_mat=zeros(c.TestSize(j),size(unique_gt,1));
            predict_mat2=zeros(c.TestSize(j),size(unique_gt,1));
            % predict and compare results
            for s=1:size(unique_gt,1)
                %use normalized distance here
                predict_mat(:,s)=histc(temp_prediction,unique_gt(s),2);
                for t=1:c.TestSize(j)
                    r=find ( temp_prediction(t,:) == unique_gt(s) & temp_prediction(t,:) >=0 );
                    if size(r,2) ~= 0
                        predict_mat2(t,s)=sum ( temp_distance ( r ) ) /size(r,2);
                    else
                        predict_mat2(t,s)=Inf;
                    end
                end
            end
            %predict_mat
            %predict_mat2
            %prediction based on 
            [max_val,final_predicted_k_result]=min(predict_mat2,[],2);    
            
            % metrics data storage%
            correct_predicted=size(find(gt_testdata==final_predicted_k_result),1);
            wrong_predicted=c.TestSize(j)-correct_predicted;
            observed_result(k,j)=correct_predicted/c.TestSize(j);
            clear temp_prediction temp_distance predict_mat predict_mat2 max_val final_predicted_k_result
        end
  
        clear training_data test_data gt_training_data gt_test_data index distance prediction
    end
    % observed_result
    % metric calculation for ith fold %
        accuracy(i,:) = mean(observed_result,2);
        
        deviation(i,:)= std(observed_result,1,2);
end
accuracy = accuracy(2:p_max,:);
deviation = deviation(2:p_max,:);