%% Yash Patel, 201301134 %
% CSE, IIIT-H %

% Open the Validation file %
f_optdigits = fopen('optdigits-orig.cv');

% Iterate over file chars, till End Of File. %
i = 1;

% Check for end of file.%
while ~feof(f_optdigits)
    
    % Store Labels, when i is divisible by 33. %
    if mod(i,33)==0
        validation_label(i/33,:)=fgetl(f_optdigits);
        
    % Store line otherwise. %
    else
        data_validation_line(i,:)=fgetl(f_optdigits);
        
    end
    
    % Increment, i and save next line. %
    i=i+1;
    
end

%Open Training file. %
f_optdigits = fopen('optdigits-orig.tra');

% Iterate over training file chars, till End of File. %
i = 1;

% Check for end of file. %
while ~feof(f_optdigits)
    
    % Store Labels, when i is divisible by 33. %
    if mod(i,33)==0
        training_label(i/33,:)=fgetl(f_optdigits);
        
    % Store line otherwise. %
    else
        data_training_line(i,:)=fgetl(f_optdigits);
    end
    
    % Increment, i and save next line. %
    i=i+1;
    
end


% Remove space before lables in label line.%
training_label = training_label(:,2);
validation_label = validation_label(:,2);

%% Char to int conversion training_data.
% From 1 and 0, convert this to binary image. %
% For 1, change indexes to int(255). %
f_optdigits = data_training_line == '1';
data_training_line(f_optdigits) = 255;

% For 0, change indexes to int(0) %
f_optdigits = data_training_line == '0';
data_training_line(f_optdigits) = 0;

%% Char to int conversion validation_data.
% From 1 and 0, convert this to binary image. %
% For 1, change indexes to int(255). %
f_optdigits = data_validation_line == '1';
data_validation_line(f_optdigits) = 255;

% For 0, change indexes to int(0) %
f_optdigits = data_validation_line == '0';
data_validation_line(f_optdigits) = 0;

%% Convert to double.
data_training_line = double(data_training_line);
data_validation_line = double(data_validation_line);

%% Question Specifications, consider data only for 7 and 0.
% Find indexes for labels 7 and 0 %
% We have a two-category system. %
f_optdigits = find(training_label == '7' | training_label == '0');

% Declare training data to be used %
train = zeros(8,9,length(f_optdigits));

for i = 1:size(f_optdigits,1)
    
    % Assign 1 for 7%
    if training_label(f_optdigits(i)) == '7'
        b = 1;
    % Assign 0 for 0%
    else
        b = 0;
    end
    
    % Make train data, line by line.%
    train(:,1:8,i) = imresize(data_training_line((f_optdigits(i)-1)*33+1:f_optdigits(i)*33-1,:),0.25);
    train(:,9,i) = b;
    
end

% Find indexes for labels 7 and 0 %
% We have a two-category system. %
f_optdigits = find(validation_label == '7' | validation_label == '0');

% Declare validation data to be used %
validate = zeros(8,9,length(f_optdigits));

for i = 1:length(f_optdigits)
    
    % Assign 1 for 7%
    if validation_label(f_optdigits(i)) == '7'
        b = 1;
    else
    % Assign 0 for 0%
        b = 0;
    end
    
    % Make validate data, line by line.%
    validate(:,1:8,i) = imresize(data_validation_line((f_optdigits(i)-1)*33+1:f_optdigits(i)*33-1,:),0.25);
    validate(:,9,i) = b;
    
end