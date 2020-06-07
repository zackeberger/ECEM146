% Parse the size of the dataset and feature vectors
num_data = 4;
num_features = 2;
    
% Initialize weight vector (dimension D) and bias
weight = [1;1];
bias = 0;

data = [2,2;-2,-4;-8,-16;3,1];
labels = [1;1;-1;-1];
    
for i = 1:1
    for j = 1:4
                    
        feature_vector = data(j, :); 
        a = dot(feature_vector, weight) + bias;
            
        % Possible update
        if labels(j)*a <= 0
            weight = weight + labels(j)*transpose(feature_vector);  % weight update
            bias = bias + labels(j);                                % bias update
        end 
            
    end
end



% Regression problem
X = [1,0;1,2;1,3];
X_T = transpose(X);
Y = [6;0;0];
    
% Obtain hat matrix H
H = inv(X_T*X)*X_T;

% Compute the weight vector
weight = H*Y