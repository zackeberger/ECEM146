% Import data
data = readtable('/Users/zackberger/Desktop/ML/HW/HW_6/data.csv');
data = data{:,:};

% Separate data into column vectors
x = data(:,[1,2]);
x1 = data(:,1);
x2 = data(:,2);
y = data(:,3);
num_data = numel(y);


% Learn the mu parameters and P(y=0)
mu_zero = zeros([2 1]);
mu_one = zeros([2 1]);
num_class_one = 0;

for i = 1 : num_data
    if y(i) == 1
        mu_one = mu_one + x(i, :).';
        num_class_one = num_class_one + 1;
    else
        mu_zero = mu_zero + x(i, :).';
    end
end

num_class_zero = num_data - num_class_one;

mu_one = (1 / num_class_one) * mu_one;
mu_zero = (1 / num_class_zero) * mu_zero;

prob_y_zero = num_class_zero / (num_class_zero + num_class_one);


% Learn the sigma parameter
sigma = zeros(2);
for i = 1 : num_data
    if y(i) == 1
        sigma = sigma + (x(i, :).' - mu_one)*(x(i, :).' - mu_one).';
    else
        sigma = sigma + (x(i, :).' - mu_zero)*(x(i, :).' - mu_zero).';
    end
end

sigma = (1/num_data) * sigma;


% Find linear decision boundary
inv_sigma = inv(sigma);

w = inv_sigma*(mu_zero - mu_one);
b = log(prob_y_zero / (1 - prob_y_zero)) + 0.5*(mu_one.'*inv_sigma*mu_one - mu_zero.'*inv_sigma*mu_zero);

hold on

% Scatter plot feature vectors
for i = 1: numel(x1)
    if y(i) == 1
        scatter(x1(i), x2(i), 'filled', 'b')
    else
        scatter(x1(i), x2(i), 'filled', 'r')
    end
end

% Plot decision boundary
x = -1:1/10000:10;
y = (-1*b - w(1)*x) / w(2);
plot(x,y, 'g');

title("GDA Visualization");
ylim([-6 2])
% xlabel("x1 feature");
% ylabel("x2 feature");

% Create legend
L(1) = plot(nan, nan, 'b:');
L(2) = plot(nan, nan, 'r:');
L(3) = plot(nan, nan, 'g:');
% legend(L, {'class 1', 'class 0', 'Decision Boundary'})


% Plot contours of the two multivariate Gaussian distributions
f = @(x,y) prob_y_zero * (1 / (2*pi)) * (1 / sqrt(det(sigma))) * exp(-0.5 * ([x;y] - mu_zero).' * inv_sigma * ([x;y] - mu_zero));
g = @(x,y) (1 - prob_y_zero) * (1 / (2*pi)) * (1 / sqrt(det(sigma))) * exp(-0.5 * ([x;y] - mu_one).' * inv_sigma * ([x;y] - mu_one));

fcontour(f, 'LevelList', logspace(-3,-1,7));
fcontour(g, 'LevelList', logspace(-3,-1,7));

hold off

title("Contour Plots of Multivariate Gaussian Distributions");