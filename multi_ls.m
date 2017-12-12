function [est_label, score] = multi_ls(train_data, train_label, test_data, kernel_type, kernel_param, lambda)
%MULTI_LS Performs LS classification using discriminants.
% training_data: a num_training_data x data_dim design matrix
% training_label: a num_training_data dimensional vector whose entry stores the
%  correspondging training label. The label should be a single numeric value.
% test_data: a num_test_data x data_dim design matrix
% kernel_type: optional. Default to Gaussian Kernel
% kernel_param: optional. Default to 3

%% Check for inputs
if ~exist('kernel_type', 'var')
   kernel_type = 'gaussian';
end

if strcmp(kernel_type, 'polynomial') && ~exist('kernel_param', 'var')
    kernel_param = 3;
elseif strcmp(kernel_type, 'gaussian') && ~exist('kernel_param', 'var')
    kernel_param = 1;
elseif ~exist('kernel_param', 'var')
    kernel_param = 1;
end

if ~exist('lambda', 'var')
    lambda = 0.1;
end

%% Standardize the inputs
standard_mean = mean(train_data, 1);
standard_std = sqrt(var(train_data, 0, 1));
train_data = bsxfun(@minus, train_data, standard_mean);
train_data = bsxfun(@rdivide, train_data, standard_std);
test_data = bsxfun(@minus, test_data, standard_mean);
test_data = bsxfun(@rdivide, test_data, standard_std);

%% Grab the necessary data
num_train_data = numel(train_label);
num_classes = max(train_label);
train_label_one_hot = zeros(num_train_data, num_classes); % Convert to one-hot encoding
train_label_one_hot(sub2ind([num_train_data, num_classes], (1:num_train_data)', train_label)) = 1;
train_label = train_label_one_hot;

%% Get the classifier
K = kerMatrix(train_data, train_data, kernel_type, kernel_param);
W = (K + lambda * eye(num_train_data))\train_label;

%% Classify
num_test_data = size(test_data,1);
score = zeros(num_test_data, num_classes);
for i=1:10000:num_test_data
    if i+10000-1 <= num_test_data
        score(i:i+10000-1,:) = kerMatrix(test_data(i:i+10000-1,:), train_data, kernel_type, kernel_param)*W;
    else
        score(i:end,:) = kerMatrix(test_data(i:end,:), train_data, kernel_type, kernel_param)*W;
    end
end;
    
[~, idx] = max(score, [], 2);

label = (1:num_classes)';
est_label = label(idx);




end

