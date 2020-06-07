% Import training data
data_train = readtable('../data/regression_train.csv');
data_train = data_train{:,:};

input_train = data_train(:,1);
output_train = data_train(:,2);

% Import testing data
data_test = readtable('../data/regression_test.csv');
data_test = data_test{:,:};

input_test = data_test(:,1);
output_test = data_test(:,2);

% Call closed-form regression algorithm
% [weight, training_error] = regressionAlg(input, output);
% weight
% training_error

% Call gradient descent algorithm
% [weight, J_w_final, iterations] = gradientAlg(input, output, 0.05);
% weight
% J_w_final
% iterations


% Call polynomial regression algorithm
deg = 2;

[weight, rms_train] = polynomialRegression(input_train, output_train, 0, deg);
weight
rms_train

[rms_test] = polynomialRegTest(input_test, output_test, weight, deg);
rms_test

hold on

% Plot data set
% scatter(input_train, output_train, 'filled', 'b')

% Plot RMS vs. Degree m
deg_number = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
train_error = [0.7622, 0.4798, 0.4609, 0.1518, 0.1487, 0.1463, 0.1454, 0.1445, 0.1343, 0.1108, 0.1114];
test_error = [0.6923, 0.5617, 0.6689, 0.1798, 0.1783, 0.1705, 0.1888, 0.4016, 1.5550, 7.6890, 8.2655];

scatter(deg_number, train_error, 'filled', 'b')
scatter(deg_number, test_error, 'filled', 'r')

% Plot best fit line
% x = 0:1/100:1;
% y = 0:1/100:1;

% plot(x,y, 'r');

title("Root Mean Square Error vs. Model Complexity");
xlabel("Degree of Polynomial");
ylabel("RMSE");

% Create legend
L(1) = plot(nan, nan, 'b');
L(2) = plot(nan, nan, 'r');
legend(L, {'Training Data', 'Testing Data'})

hold off
