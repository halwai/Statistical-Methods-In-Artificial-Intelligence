%poker hands data
%preprocessing 
file_id=fopen('poker-hand-training-true.data.txt');
c=textscan(file_id,'%d %d %d %d %d %d %d %d %d %d %d','delimiter',',');
fclose(file_id);

data=zeros(25010,10);
for i=1:10
    data(:,i)=c{i};
end

results=c{11};

%results on - folfd knn  using the custom myknn function
[mean,deviation]=myknn(data,results,5,5);

%draw plots
for i=1:4
figure
errorbar( mean(i,:) , deviation(i,:) , ':bs');
title (strcat('data-poker-hands  p-fold p=',num2str(i+1)) );
end
% mean
% display(strcat('Min :    ',num2str(min(mean(:)))))
% display(strcat('Max :    ',num2str(max(mean(:)))))
% display(strcat('Average :',num2str(sum(mean(:)/20))))

clear c data results
