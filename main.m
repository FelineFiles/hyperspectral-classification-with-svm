%% Clear the workspace
close all
fclose all;
clear
clc

%% Plot the images by calling read_classes.m
read_classes
clear;

%% Parameters to control the program
filter_type = 'No';         % Filter used to denoise the image. Can be 'Wiener', 'Median', 'Box', or 'None'
use_lidar = false;            % Use altitude information as a feature (does not improve anything)
use_entropy = false;          % Use entropy as a feature
do_pca = false;                % Use PCA to do feature compreesion
num_components_kept = 9;     % Number of components used in PCA. Setting it too low results in long training time. 
do_cv = true;                 % Performs cross validation to assess how good our parameters are

svm_kernel_type = 'linear'; % The kernel used by SVM to make the actual estimation, can be either 'gaussian' or 'polynomial'
svm_kernel_param = 1;         % Kernel parameter. When kernel type is 'gaussian', this becomes polynomial order 
ls_kernel_type = 'gaussian';  % Like svm_kernel_type but for LS classifier
ls_kernel_param = 1;          % Like svm_kernel_param but for LS classifier

type = 'none_nolidar_noentropy_pca';
save_path = ['./figures/', type, '/'];
mkdir(save_path);
wait_time = 2;

%% Read the data
load 2013_DFC_contest.mat
hyper = double(hyper);
[height, width, num_channels] = size(hyper);

%% Some parameters 
file_dir = './ground_truth/';
classes = {'grass_healthy', 'grass_stressed', 'grass_synthetic', 'tree','soil',...
    'water', 'residential', 'commercial', 'road', 'highway', ...
    'railway', 'parkinglot1', 'parkinglot2', 'tennis_court', 'running_track'};
class_names = {'grass healthy', 'grass stressed', 'grass synthetic', 'tree','soil',...
    'water', 'residential', 'commercial', 'road', 'highway', ...
    'railway', 'parking lot 1', 'parking lot 2', 'tennis court', 'running track', ...
    'unknown'};
file_extension = '.txt';
num_classes = numel(classes);

color_vect = [0,205,0; 127,255,0; 46,139,8; 0,139,0; 160,82,45;...
              0,255,255; 255,255,255; 216,191,216; 255,0,0; 139,0,0;...
              60,0,0; 255,255,0; 238,154,0; 85,26,139; 255,127,80; ...
              0,0,0]; % colors of the different classes

rgb_channels = [57,30,20];

%% Denoise the image with filter
mask_size = [7,7];

filtered_hyper = hyper;

% Wiener Filter
if strcmp(filter_type, 'Wiener')
    for c=1:num_channels
       filtered_hyper(:,:,c) = wiener2(hyper(:,:,c), mask_size); 
    end
end

% Median Filter
if strcmp(filter_type, 'Median')
    mask_size = [7,7];
    for c=1:num_channels
       filtered_hyper(:,:,c) = medfilt2(hyper(:,:,c), mask_size); 
    end
end

% Box Filter
if strcmp(filter_type, 'Box')
    for c=1:num_channels
       filtered_hyper(:,:,c) = imboxfilt2(hyper(:,:,c), mask_size); 
    end
end

%% Plot the filtered data
figure
rgb_full(:,:,1) = imadjust(rescale(double(filtered_hyper(:,:,rgb_channels(1))),1)); % show color image
rgb_full(:,:,2) = imadjust(rescale(double(filtered_hyper(:,:,rgb_channels(2))),1));
rgb_full(:,:,3) = imadjust(rescale(double(filtered_hyper(:,:,rgb_channels(3))),1));
f = imshow(rgb_full,[]);
title(['RGB of Image Filtered with ', filter_type, ' Filter']);
pause(wait_time);
frame_h = get(handle(gcf),'JavaFrame');
set(frame_h,'Maximized',1);
set(gcf, 'PaperPositionMode', 'auto');
pause(wait_time);
saveas(f, [save_path, 'filtered.png']);
clear rgb_full;

%% Get the training data
training_pixel = [];
training_label = [];
training_label_one_hot = [];

vec_hyper = reshape(filtered_hyper, [height*width, num_channels]);
vec_lidar = reshape(lidar, [height*width, 1]);
if use_lidar
    vec_hyper = [vec_hyper, vec_lidar];
end
if use_entropy
   entropy = entropyfilt(filtered_hyper);
   vec_entropy = reshape(entropy, height*width, []);
   vec_hyper = [vec_hyper, vec_entropy];
end

for k=1:num_classes
    % read in the location of the pixels for each class
    tt = textscan(fopen([file_dir, classes{k}, file_extension]),'%d%d%d%f%f'); 

    num_class_pts = numel(tt{1});
    training_pixel = [training_pixel; vec_hyper(sub2ind([height,width], tt{3}, tt{2}), :)];
    training_label = [training_label; k*ones(num_class_pts,1)];
end

%% Standardize the data (important, else it underflows gaussian kernel)
standard_mean = mean(training_pixel, 1);
standard_std = sqrt(var(training_pixel, 0, 1));
training_pixel = bsxfun(@minus, training_pixel, standard_mean);
training_pixel = bsxfun(@rdivide, training_pixel, standard_std);
vec_hyper = bsxfun(@minus, vec_hyper, standard_mean);
vec_hyper = bsxfun(@rdivide, vec_hyper, standard_std);

%% Use PCA to visualize the filtered data
[coeff, score, latent, ~, explained] = pca(training_pixel);

figure;
hold on;
for i=1:4
    f = plot(coeff(:,i));
end
xlabel('Spectral Channel');
ylabel('Weight');
title('First Four Principal Components');
legend('Component 1', 'Component 2', 'Component 3', 'Component 4');
% pause(wait_time);
% frame_h = get(handle(gcf),'JavaFrame');
% set(frame_h,'Maximized',1)
% set(gcf, 'PaperPositionMode', 'auto');
% pause(wait_time);
saveas(f, [save_path, 'componenets.png']);

figure;
f = plot(explained);
xlabel('Number of Components');
ylabel('Perecent Explained (%)');
title('Percentage of Variance Explained by Each Component');
xlim([1,20]);
pause(wait_time);
% frame_h = get(handle(gcf),'JavaFrame');
% set(frame_h,'Maximized',1)
% set(gcf, 'PaperPositionMode', 'auto');
% pause(wait_time);
saveas(f, [save_path, 'explained.png']);

figure;
plot(0:numel(explained), cumsum([0;explained]));
xlabel('Number of Components');
ylabel('Perecent Explained (%)');
title('Percentage of Total Variance Explained');
xlim([0,20]);
ylim([0,100]);
% pause(wait_time);
% frame_h = get(handle(gcf),'JavaFrame');
% set(frame_h,'Maximized',1);
% set(gcf, 'PaperPositionMode', 'auto');
% pause(wait_time);
saveas(f, [save_path, 'total_explained.png']);

figure;
hold on;
p = zeros(num_classes,1);
for k=1:num_classes
    class_idx = find(training_label == k);
    p(k) = scatter3(score(class_idx, 1), score(class_idx, 2), score(class_idx, 3), '.');
end
legend(p, class_names);
xlabel('Component 1');
ylabel('Component 2');
zlabel('Component 3');
title('Projection of Training Pixels onto the First Three Principal Components');
pause(wait_time);
frame_h = get(handle(gcf),'JavaFrame');
set(frame_h,'Maximized',1);
set(gcf, 'PaperPositionMode', 'auto');
pause(wait_time);
saveas(p(end), [save_path, 'projection.png']);

%% Select the number of components used for analysis
if do_pca
    training_data = score(:,1:num_components_kept);

    hyper_score = vec_hyper * coeff;
    test_data = hyper_score(:, 1:num_components_kept);
else
    training_data = training_pixel;
    test_data = vec_hyper;
end

%% Performs cross validation to find the best parameter
if do_cv
    [svm_stat, ls_stat] = cross_validate(training_data, training_label, 5, true);
    ls_stat.gauss_err
    %ls_stat.poly_err
    svm_stat.gauss_err
    %svm_stat.poly_err
    save([save_path, 'cv.mat'], 'svm_stat', 'ls_stat');
end
    
%% Make Prediction
fprintf('Estimating Classes with SVM...\n');
[label_svm, score] = multi_svm(training_data, training_label, test_data, svm_kernel_type, svm_kernel_param);
fprintf('Done!\n');
fprintf('Estimating Classes with LS...\n');
label_ls = multi_ls(training_data, training_label, test_data, ls_kernel_type, ls_kernel_param);
fprintf('Done!\n');

%% Plot the prediction
% Plot the prediction from SVM
prediction_svm = im2uint8(zeros(height*width, 3));
for k=1:num_classes
    idx = label_svm==k;
    prediction_svm(find(label_svm==k),:) = repmat(color_vect(k,:), [sum(idx),1]);
end
prediction_svm = reshape(prediction_svm, [height, width, 3]);

figure; f = imshow(prediction_svm,[]);
title('Prediction with SVM');
add_class_legend;
pause(wait_time);
frame_h = get(handle(gcf),'JavaFrame');
set(frame_h,'Maximized',1);
set(gcf, 'PaperPositionMode', 'auto')
pause(wait_time);
saveas(f, [save_path, 'svm.png']);

% Plot the prediction from LS
prediction_ls = im2uint8(zeros(height*width, 3));
for k=1:num_classes
    idx = label_ls==k;
    prediction_ls(find(label_ls==k),:) = repmat(color_vect(k,:), [sum(idx),1]);
end
prediction_ls = reshape(prediction_ls, [height, width, 3]);

figure, f = imshow(prediction_ls,[]);
title('Prediction with LS');
add_class_legend;
pause(wait_time);
frame_h = get(handle(gcf),'JavaFrame');
set(frame_h,'Maximized',1)
set(gcf, 'PaperPositionMode', 'auto');
pause(wait_time);
saveas(f, [save_path, 'ls.png']);

%% Plot the result after mode filtering
% Plot the prediction from SVM with mode filter
label_svm_smooth = colfilt(reshape(label_svm, [height, width]) ,[3, 3],'sliding',@mode);
label_svm_smooth = reshape(label_svm_smooth, [height*width,1]);
prediction_svm = reshape(prediction_svm, [height*width, 3]);
for k=1:num_classes
    idx = label_svm_smooth==k;
    prediction_svm(find(idx),:) = repmat(color_vect(k,:), [sum(idx),1]);
end
prediction_svm = reshape(prediction_svm, [height, width, 3]);
figure; f = imshow(prediction_svm,[]);
title('Prediction with SVM and Mode Filter');
add_class_legend;
pause(wait_time);
frame_h = get(handle(gcf),'JavaFrame');
set(frame_h,'Maximized',1);
set(gcf, 'PaperPositionMode', 'auto');
pause(wait_time);
saveas(f, [save_path, 'svm_mode.png']);

% Plot the prediction from LS with mode filter
label_ls_smooth = colfilt(reshape(label_ls, [height, width]) ,[3, 3],'sliding',@mode);
label_ls_smooth = reshape(label_ls_smooth, [height*width, 1]);
prediction_ls = reshape(prediction_ls, [height*width, 3]);
for k=1:num_classes
    idx = label_ls_smooth==k;
    prediction_ls(find(idx),:) = repmat(color_vect(k,:), [sum(idx),1]);
end
prediction_ls = reshape(prediction_ls, [height, width, 3]);
figure; f = imshow(prediction_ls,[]);
title('Prediction with LS and Mode Filter');
add_class_legend;
pause(wait_time);
frame_h = get(handle(gcf),'JavaFrame');
set(frame_h,'Maximized',1);
set(gcf, 'PaperPositionMode', 'auto');
pause(wait_time);
saveas(f, [save_path, 'ls_mode.png']);
