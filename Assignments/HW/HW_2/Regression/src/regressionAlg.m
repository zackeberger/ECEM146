function [weight, training_error] = regressionAlg(input,output)
%regressionALg implements the closed-form solution for linear regression

    % Obtain the matrix X
    [num_input, ~] = size(input);
    one_matrix = ones(num_input,1);
    X = [one_matrix, input];
    
    % Obtain hat matrix H
    X_T = transpose(X);
    H = inv(X_T*X)*X_T;

    % Compute the weight vector
    weight = H*output;
    
    % Compute error relative to training data
    training_error = dot(X*weight - output, X*weight - output);
    
end
