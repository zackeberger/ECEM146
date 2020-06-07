function [weight, J_w_final, iterations] = gradientAlg(input, output, step)
% gradientAlg implements the gradient descent algorithm on 1 dimensional
% data as provided in HW_2 for M146 @ UCLA
    
    % Obtain the matrix X (Nx2 dimensional)
    [num_input, ~] = size(input);
    one_matrix = ones(num_input,1);
    X = [one_matrix, input];
    
    % Initialize weight vector, dimension of 2x1 to account for bias w0
    weight = zeros(2,1);
    
    % Perform gradient descent for 10000 iterations
    iter = 0;
    for i = 1:10000
        
        % Compute the training error J(w) based on previously computed
        % weight vector
        J_old = dot(X*weight - output, X*weight - output);
        
        % Compute the gradient at the current weight vector
        gradient = zeros(2,1);
        for j = 1:num_input
            gradient = gradient + (weight(1) + weight(2)*input(j) - output(j))*[1;input(j)];
        end
        
        % Update the weight vector
        weight = weight - step*gradient;

        % Compute the training error J(w) based on freshly computed
        % weight vector     
        J_new = dot(X*weight - output, X*weight - output);
        
        iter = iter + 1;
        
        if abs(J_old - J_new) < 0.0001
            break  
        end
                 
    end

    iterations = iter;
    J_w_final = J_new;

end
