function K=kerMatrix(X_1, X_2, kernal_type, kernel_param)
% If Kernel is Guassian, param is variance. If kernel is polynomial, param is variance 

if strcmp(kernal_type, 'linear')
    K = X_1 * X_2';
elseif strcmp(kernal_type, 'polynomial')
    if ~exist('kernel_param', 'var')
        kernel_param = 3;
    end
    K=(1 + X_1 * X_2').^kernel_param;
elseif strcmp(kernal_type, 'gaussian')
    if ~exist('kernel_param', 'var')
        kernel_param = 1;
    end
    X_1_sq = sum(X_1.^2,2);
    X_2_sq = sum(X_2.^2,2);
    n1 = size(X_1, 1);
    n2 = size(X_2, 1);

    D = (X_1_sq * ones(1, n2)) + (ones(n1,1)*X_2_sq') - 2*X_1*X_2';

    K = exp(-D/kernel_param);
end

    