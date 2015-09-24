%tic-tac-toe data
%preprocessing 
file_id=fopen('tic-tac-toe.data.txt');
c=textscan(file_id,'%c %c %c %c %c %c %c %c %c %s','delimiter',',');
fclose(file_id);

data=zeros(958,9);
for i=1:9
    data(:,i)=c{i};
end
data ( data =='x')=1;
data ( data =='o')=2;
data ( data =='b')=3;

results=c{10};
results ( find ((strcmp(results,'positive'))==1))=mat2cell(['1']);
results ( find ((strcmp(results,'negative'))==1))=mat2cell(['2']);

results=str2num(cell2mat(results));

%results on - folfd knn  using the custom myknn function
[mean,deviation]=myknn(data,results,5,5);

%draw plots
for i=1:4
figure
errorbar( mean(i,:) , deviation(i,:) , ':bs');
title (strcat('data-tic-tac-toe  p-fold p=',num2str(i+1)) );
end
% mean
% display(strcat('Min :    ',num2str(min(mean(:)))))
% display(strcat('Max :    ',num2str(max(mean(:)))))
% display(strcat('Average :',num2str(sum(mean(:)/20))))

clear c data results