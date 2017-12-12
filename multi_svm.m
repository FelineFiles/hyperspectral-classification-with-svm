function [est_label, score] = multi_svm(training_data, training_label, test_data, kernel_type, kernel_param, verbose )
%MULTI_SVM Performs one vs all multi-class classification.
% training_data: a num_training_data x data_dim design matrix
% training_label: a num_training_data vector whose entry stores the
%  correspondging training label. The label should be a single numeric value.
% test_data: a num_test_data x data_dim design matrix
% kernel_type: optional. Default to Gaussian Kernel
% kernel_param: optional. Default to 3

%% Get the inputs
[num_training_data, data_dim] = size(training_data);
num_test_data = size(test_data, 1);
classes = unique(training_label);
num_classes = numel(classes);
score = nan(num_test_data, num_classes);
svms = cell(num_classes,1);
box_constraint = 10;

if ~exist('kernel_type', 'var')
   kernel_type = 'gaussian';
end

if strcmp(kernel_type, 'polynomial') && ~exist('kernel_param', 'var')
    kernel_param = 3;
end

if ~exist('verbose', 'var')
    verbose = 0;
end

% %% Standardize the inputs
% standard_mean = mean(training_data, 1);
% standard_std = sqrt(var(training_data, 0, 1));
% training_data = bsxfun(@minus, training_data, standard_mean); 
% training_data = bsxfun(@rdivide, training_data, standard_std);
% test_data = bsxfun(@minus, test_data, standard_mean);
% test_data = bsxfun(@rdivide, test_data, standard_std);

%% Train the svms
for k=1:num_classes
    if verbose
        fprintf(['Training Classifier ', num2str(classes(k)) ' of ', num2str(num_classes), '\n']);
    end
    class_k_label = training_label == classes(k);
    if strcmp(kernel_type, 'polynomial')
        svms{k} = fitcsvm(training_data, class_k_label, 'Standardize', true, ...
            'KernelScale', 'auto', 'KernelFunction', kernel_type, ...
            'PolynomialOrder',kernel_param, 'CacheSize', 'maximal', 'BoxConstraint', box_constraint);
    else
        svms{k} = fitcsvm(training_data, class_k_label, 'Standardize', true, ...
            'KernelScale', 'auto', 'KernelFunction', kernel_type, 'CacheSize', 'maximal', 'BoxConstraint', box_constraint);
    end
end;

%% Classify the test data
for k=1:num_classes
    if verbose
        fprintf(['Classifying with Classifier ', num2str(classes(k)) ' of ', num2str(num_classes), '\n']);
    end
    [~, temp_score] = predict(svms{k}, test_data);
    score(:, k) = temp_score(:, 2);
end;

[~, est_label] = max(score, [], 2);

clear svms


end

