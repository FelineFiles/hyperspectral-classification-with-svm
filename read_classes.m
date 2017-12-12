%% Clear the workspace
close all
clear
clc

%% Read the data
load 2013_DFC_contest.mat  
[height, width, num_channels] = size(hyper);

%% Some parameters for plotting
rgb_channels = [57,30,20]; % RGB channels

file_dir = './ground_truth/';
classes = {'grass_healthy', 'grass_stressed', 'grass_synthetic', 'tree',...
    'soil', 'water', 'residential', 'commercial', 'road', 'highway', ...
    'railway', 'parkinglot1', 'parkinglot2', 'tennis_court', 'running_track'};
file_extension = '.txt';
num_classes = numel(classes);

color_vect = [0, 205, 0;127, 255, 0;46, 139, 8;0, 139, 0;160, 82, 45;...
    0, 255, 255;255, 255, 255;216, 191, 216;255, 0, 0;139, 0, 0;60, 0, 0;...
    255, 255, 0;238, 154, 0;85, 26, 139;255, 127, 80]; % colors of the different classes

%% Plot the RGB and Lidar
% Plot the RGB Image
figure
rgb_full(:,:,1) = imadjust(rescale(double(hyper(:,:,rgb_channels(1))),1)); % show color image
rgb_full(:,:,2) = imadjust(rescale(double(hyper(:,:,rgb_channels(2))),1));
rgb_full(:,:,3) = imadjust(rescale(double(hyper(:,:,rgb_channels(3))),1));
imshow(rgb_full,[]);
title('RGB Image');

% Plot the lidar image
figure, imshow(lidar,[],'colormap', jet) % show lidar dataset (measures the elevation (in meters) of the different structures in the image). This can be used as an additional feature.
title('Lidar (Elevation)');

%% Plot the training sample for each class
% For plotting
rgb_int = im2uint8(rgb_full); 
ground_truth = uint8(zeros(size(rgb_full)));
rgb_int = reshape(rgb_int, [height*width, 3]);
ground_truth = reshape(ground_truth, [height*width, 3]);

% the .txt files containing the training samples are organized this way:
% each row represents one training point
% first column: sample #
% second and third column: vertical and horizontal coordinates of the sample
% fourth and fifth colums: latitude and longitude of the sample (not used
% here)

for k=1:num_classes
    % read in the location of the pixels for each class
    tt = textscan(fopen([file_dir, classes{k}, file_extension]),'%d%d%d%f%f'); 

    num_pts = numel(tt{1});
    rgb_int(sub2ind([height,width], tt{3}, tt{2}), :) = repmat(color_vect(k,:), [num_pts, 1]);
    ground_truth(sub2ind([height,width], tt{3}, tt{2}), :) = repmat(color_vect(k,:), [num_pts, 1]);
end

rgb_int = reshape(rgb_int, [height, width, 3]);
ground_truth = reshape(ground_truth, [height, width, 3]);

% Show resulting images
figure, imshow(rgb_int,[]);
title('RGB Image with Highlighted Training Samples');
add_class_legend;
figure, imshow(ground_truth,[])
title('Training Samples');
add_class_legend;