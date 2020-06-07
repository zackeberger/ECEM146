function [weight, bias, num_updates] = perceptronAlg(data,labels)
%perceptronAlg implements the Perceptron ML algorithm

    % Parse the size of the dataset and feature vectors
    [num_data, num_features] = size(data);
    
    % Initialize weight vector (dimension D) and bias
    weight = zeros(1,num_features);
    bias = 0;
    num_updates = 0;
    
    for i = 1:1000
        for j = 1:num_data
                    
            feature_vector = data(j, :);    
            a = dot(feature_vector, weight) + bias;
            
            % Possible update
            if labels(j)*a <= 0
                weight = weight + labels(j)*feature_vector;  % weight updtae
                bias = bias + labels(j);                     % bias update
                num_updates = num_updates + 1;
            end 
            
        end
    end
end

