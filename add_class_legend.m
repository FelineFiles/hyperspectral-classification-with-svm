function  add_class_legend()
%ADD_CLASS_LEGEND Add a color legend for the classes to the current image
%% Color and name of the classes
color_vect = [0,205,0; 127,255,0; 46,139,8; 0,139,0; 160,82,45;...
              0,255,255; 255,255,255; 216,191,216; 255,0,0; 139,0,0;...
              60,0,0; 255,255,0; 238,154,0; 85,26,139; 255,127,80; ...
              0,0,0];
class_names = {'grass healthy', 'grass stressed', 'grass synthetic', 'tree','soil',...
    'water', 'residential', 'commercial', 'road', 'highway', ...
    'railway', 'parking lot 1', 'parking lot 2', 'tennis court', 'running track'};
num_classes = numel(class_names);

%% Add the legend
hold on;
p = zeros(num_classes, 1);
for k=1:num_classes;
    p(k) = plot(nan, nan, 's', 'MarkerFaceColor', color_vect(k,:)/255, 'MarkerEdgeColor', 'k');
end
legend(p, class_names);
hold off;



end

