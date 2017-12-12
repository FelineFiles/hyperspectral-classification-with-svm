function [svm_stat, ls_stat] = cross_validate(data, label, K, visualize )
%CROSS_VALIDATE Performs K-fold cross validation with polynomial kernel and
%gaussian kernel
% sv_stat and ls_stat are two structs storing the error rate and the
% confusion matrix
%% Get the basic variables
num_data = size(data,1);
num_classes = max(label);
max_poly_order = 4;
poly_sv_err = nan(max_poly_order,1);
poly_ls_err = nan(max_poly_order,1);
poly_range = 1:max_poly_order;

if ~exist('full', 'var')
    full = true;
end

if ~exist('visualize', 'var')
    visualize = false;
end
visualize = true;
% For confusion matrix
class_names = {'grass healthy', 'grass stressed', 'grass synthetic', 'tree','soil',...
    'water', 'residential', 'commercial', 'road', 'highway', ...
    'railway', 'parking lot 1', 'parking lot 2', 'tennis court', 'running track'};  

%% Prepare indices for cross validation
rng(1);
cv_idx = [];
for k=1:num_classes
    cv_idx = [cv_idx; crossvalind('Kfold', sum(label==k), 5)];
end

%% Get the vector of actual labels so we can calculate confusion matrix later
actual_label = [];
for k=1:K
    test = (cv_idx == k);
    actual_label = [actual_label; label(test)];
end

%% Try LS with Gaussian Kernel
err = nan(K,1);
ls_gauss_pred = [];
for k=1:K
    fprintf(['Cross Validation: LS with Gaussian Kernel: Fold ', num2str(k) ' of ', num2str(K), '\n']);
    test = (cv_idx == k);
    train = ~test;
    label_hat = multi_ls(data(train, :), label(train), data(test,:), 'gaussian'); 
    err(k) = 100*mean(label_hat ~= label(test));
    
    ls_gauss_pred = [ls_gauss_pred; label_hat];
end
ls_stat.gauss_err = mean(err);
[ls_stat.gauss_conf, ls_stat.gauss_conf_class] = confusionmat(class_names(actual_label), class_names(ls_gauss_pred));


%% Next try LS with Polynomial Kernel
ls_stat.poly_conf = cell(numel(poly_range),1);
ls_stat.poly_conf_class = cell(numel(poly_range),1);
for i=1:numel(poly_range)
    ls_poly_pred = [];
    for k=1:K
        fprintf(['Cross Validation: LS with Order ', num2str(poly_range(i)), ' Polynomial Kernel: Fold ', num2str(k) ' of ', num2str(K), '\n']);
        test = (cv_idx == k);
        train = ~test;
        label_hat = multi_ls(data(train, :), label(train), data(test,:), 'polynomial', poly_range(i)); 
        err(k) = 100*mean(label_hat ~= label(test));
        
        ls_poly_pred = [ls_poly_pred; label_hat];
    end
    poly_ls_err(i) = mean(err); 
    [ls_stat.poly_conf{i}, ls_stat.poly_conf_class{i}] = confusionmat(class_names(actual_label), class_names(ls_poly_pred));
end
ls_stat.poly_err = poly_ls_err;


%% Try SVM with gaussian kernel
err = nan(K,1);
svm_gauss_pred = [];
for k=1:K
    fprintf(['Cross Validation: SVM with Gaussian Kernel: Fold ', num2str(k) ' of ', num2str(K), '\n']);
    test = (cv_idx == k);
    train = ~test;
    label_hat = multi_svm(data(train, :), label(train), data(test,:), 'gaussian', [],  false); 
    err(k) = 100*mean(label_hat ~= label(test));
    
    svm_gauss_pred = [svm_gauss_pred; label_hat];
end
svm_stat.gauss_err = mean(err);
[svm_stat.gauss_conf, svm_stat.gauss_conf_class] = confusionmat(class_names(actual_label), class_names(svm_gauss_pred));

% figure;
% actual_label_one_hot = zeros(size(actual_label,1), num_classes); % Convert to one-hot encoding
% actual_label_one_hot(sub2ind([size(actual_label,1), num_classes], (1:size(actual_label,1))', actual_label)) = 1;
% svm_gauss_pred_one_hot = zeros(size(actual_label,1), num_classes); % Convert to one-hot encoding
% svm_gauss_pred_one_hot(sub2ind([size(actual_label,1), num_classes], (1:size(actual_label,1))', svm_gauss_pred)) = 1;
% plotconfusion(actual_label_one_hot', svm_gauss_pred_one_hot', 'Gaussian SVM + PCA');
% xlabel('Predicted Class');
% ylabel('Target Class');
% set(gca, 'xticklabel', class_names);
% set(gca, 'yticklabel', class_names);

%% Next try SVM polynomial kernel
svm_stat.poly_conf = cell(numel(poly_range),1);
svm_stat.poly_conf_class = cell(numel(poly_range),1);
for i=1:numel(poly_range)
    svm_poly_pred = [];
    for k=1:K
        fprintf(['Cross Validation: SVM with Order ', num2str(poly_range(i)), ' Polynomial Kernel: Fold ', num2str(k) ' of ', num2str(K), '\n']);
        test = (cv_idx == k);
        train = ~test;
        label_hat = multi_svm(data(train, :), label(train), data(test,:), 'polynomial', poly_range(i), false); 
        err(k) = 100*mean(label_hat ~= label(test));
        
        svm_poly_pred = [svm_poly_pred; label_hat];
    end
    poly_sv_err(i) = mean(err); 
    [svm_stat.poly_conf{i}, svm_stat.poly_conf_class{i}] = confusionmat(class_names(actual_label), class_names(svm_poly_pred));
end
svm_stat.poly_err = poly_sv_err;

%% Plot the result
if visualize
    figure;
    hold on;
    plot(ls_stat.poly_err);
    plot(ls_stat.gauss_err * ones(max_poly_order, 1));
    plot(svm_stat.poly_err);
    plot(svm_stat.gauss_err * ones(max_poly_order, 1));
    ylabel('Error Rate (%)');
    xlabel('Polynomial Order');
    title('Error Rate of Classifier with Different Kernels');
    legend('LS: Polynomial', 'LS: Gaussian', 'SVM: Polynomial', 'SVM: Gaussian', 'Location', 'best');
end

end

