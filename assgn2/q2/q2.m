clc
clear 

w1 = [ 1 7 ;  6 3;  7 8 ;  8 9 ; 4 5 ;  7 5];
w2 = [ 3 4 ; 4 3 ;  2 4 ;  7 1 ; 1 3 ;  4 2];

x=[ 1  1 7 ; 1 6 3; 1 7 8 ; 1 8 9 ;1 4 5 ; 1 7 5 ;...
     -1 -3 -4 ;-1 -4 -3 ;-1 -2 -4 ; -1 -7 -1; -1 -1 -3 ; -1 -4 -2];

[m,d]=size(x);
theta=zeros([d ,1]); %zeros initalization of weight vectors
 
% part a
figure;
plot(w1(:,1)+i*w1(:,2),'or','MarkerSize',10);
hold on;
plot(w2(:,1)+i*w2(:,2),'xb','MarkerSize',10);

t1=single_sample_perceptron(x,theta);
t2=single_sample_perceptron_margin(x,theta,0.5);
t3=single_sample_perceptron_relaxation_margin(x,theta,0.5);
t4=LMS(x,theta,0.03);

plot([0,t1(2)]+[0,i*t1(3)],'--vg');
plot([0,t2(2)]+[0,i*t2(3)],'--<y');
plot([0,t3(2)]+[0,i*t3(3)],'--^b');
plot([0,t4(2)]+[0,i*t4(3)],'-->m');
legend('w1','w2','single-sample-perceptron',...
    'single-sample-perceptron-margin',...
    'single-sample-perceptron-margin-relaxation','LMS');

hold off

% part b
theta_set = [ t1' ; t2' ; t3' ; t4';...
             %optimal directions
             0 1 0; 0 1 1 ; 0 0 1;  0 -1 1 ;...
             0 -1 0 ; 0 -1 -1 ; 0 0 -1 ;  0 -1 1 ; ...
             %different directions in 2d space
             0 100 0; 0 100 100 ; 0 0 100;  0 -100 100 ;...
             0 -100 0 ; 0 -100 -100 ; 0 0 -100 ;  0 -100 100 ; ... ...  
             %different directions in 2d space with multipliers
             0 -10 1000 ;0 100 50 ;...
             % some random numbers
             10 1 1 ; 100 1 1 ; -10 1 1 ; -100 1 1]; 
             % changing bias
time=zeros(4,length(theta_set));
b=0.3; % margin
for i=1:length(theta_set)
    tic;
    single_sample_perceptron(x,theta_set(i,:)');
    time(1,i)=toc;
end
for i=1:length(theta_set)
    tic;
    single_sample_perceptron_margin(x,theta_set(i,:)',b);
    time(2,i)=toc;
end
for i=1:length(theta_set)
    tic;
    single_sample_perceptron_relaxation_margin(x,theta_set(i,:)',b);
    time(3,i)=toc;
end
for i=1:length(theta_set)
    tic;
    LMS(x,theta_set(i,:)',b);
    time(4,i)=toc;
end
for i=1:4
    figure;
    plot(time(i,:)); 
    legend(
            't1' ; 't2' ; 't3' ; 't4';...
             %optimal directions
             '0 1 0'; '0 1 1' ; '0 0 1'; '0 -1 1' ;...
             '0 -1 0' ; '0 -1 -1' ; '0 0 -1' ;  '0 -1 1' ; ...
             %different directions in 2d space
             '0 100 0'; '0 100 100' ; '0 0 100';  '0 -100 100' ;...
             '0 -100 0' ; '0 -100 -100' ; '0 0 -100' ;  '0 -100 100' ; ... ...  
             %different directions in 2d space with multipliers
             '0 -10 1000' ;'0 100 50' ;...
             % some random numbers
             '10 1 1' ; '100 1 1' ; '-10 1 1' ; '-100 1 1');

end

%part c
b=[0.1 0.5 1  5  10 ];
temp =['--vg';'-->b';'--<r';'--^y';'--*m'];
time=zeros(2,length(b));
figure;
hold on;
for i=1:length(b)
    tic;
    t1=single_sample_perceptron_margin(x,theta,b(i));
    time(1,i)=toc;
    plot([0,t1(2)]+[0,i*t1(3)],temp(i,:));
end
hold off;
figure;
hold on;
for i=1:length(b)
    tic;
    t2=single_sample_perceptron_relaxation_margin(x,theta,b(i));
    time(2,i)=toc;
    plot([0,t2(2)]+[0,i*t2(3)],temp(i,:));
end
hold off;

for i=1:2
    figure;
    plot(time(i,:)); 
end
