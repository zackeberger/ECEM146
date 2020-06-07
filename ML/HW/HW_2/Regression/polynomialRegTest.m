function [rms] = polynomialRegTest(input, output, weight, degree)
%polynomialRegTest implement returns the J(w) error for a dataset and
%weight vector

    % Obtain the matrix PHI
    [num_input, ~] = size(input);
    deg_0 = ones(num_input,1);
    PHI = [deg_0];
    
    % Iteratively add on columns to PHI based on degree
    for i = 1:degree
        deg_i = input.^i;
        PHI = [PHI, deg_i];
    end
    
    % Obtain hat matrix H
    PHI_T = transpose(PHI);
    H = inv(PHI_T*PHI)*PHI_T;
    
    % Compute the root mean square error
    J = dot(PHI*weight - output, PHI*weight - output);
    rms = sqrt(J / num_input);
    
end

