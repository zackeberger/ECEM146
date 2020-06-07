% Import data
data = readtable('../data/data2.csv');
data = data{:,:};


% Separate data into column vectors
x = data(:,[1,2]);
x1 = data(:,1);
x2 = data(:,2);
y = data(:,3);

hold on

% Run Perceptron and plot decision boundary
[weight, bias, num_updates] = perceptronAlg(x,y);
normalized_weight = weight / sqrt(weight(1)*weight(1) + weight(2)*weight(2));

% Find the margin
[num_data, ~] = size(x);
margin = 200000000;
for j = 1:num_data
                    
    feature_vector = x(j, :); 
    a = dot(feature_vector, normalized_weight) + bias;
    a = y(j) * a;
    
    if a < margin || margin == 200000000
        margin = a;
    end
        
end

margin

% Scatter plot feature vectors
for i = 1: numel(x1)
    if y(i) == 1
        scatter(x1(i), x2(i), 'filled', 'b')
    else
        scatter(x1(i), x2(i), 'filled', 'r')
    end
end

% Plot decision boundary
x = -1:1/100:1;
y = (bias - weight(1)*x) / weight(2);
plot(x,y, 'g');

title("Data 3");
xlabel("x1 feature");
ylabel("x2 feature");

% Create legend
L(1) = plot(nan, nan, 'b:');
L(2) = plot(nan, nan, 'r:');
L(3) = plot(nan, nan, 'g:');
legend(L, {'class 1', 'class -1', 'decision boundary'})

hold off
