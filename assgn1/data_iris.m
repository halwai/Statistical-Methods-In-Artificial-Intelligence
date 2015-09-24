%iris data
%preprocessing 
file_id=fopen('Iris.data.txt');
c=textscan(file_id,'%f %f %f %f %s','delimiter',',');
fclose(file_id);

data=zeros(150,4);
for i=1:4
    data(:,i)=c{i};
end

results=c{5};
results ( find ((strcmp(results,'Iris-setosa'))==1))= mat2cell(['1']);
results ( find ((strcmp(results,'Iris-versicolor'))==1))= mat2cell(['2']);
results ( find ((strcmp(results,'Iris-virginica'))==1))= mat2cell(['3']);

results=str2num(cell2mat(results));


%results on - folfd knn  using the custom myknn function
[mean,deviation]=myknn(data,results,5,5);

%draw plots
for i=1:4
figure
errorbar( mean(i,:) , deviation(i,:) , ':bs');
title (strcat('data-iris  p-fold p=',num2str(i+1)) );
end
% mean
% display(strcat('Min :    ',num2str(min(mean(:)))))
% display(strcat('Max :    ',num2str(max(mean(:)))))
% display(strcat('Average :',num2str(sum(mean(:)/20))))

clear c data results
    