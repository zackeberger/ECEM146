% Import data
data = readtable('/Users/zackberger/Desktop/ML/HW/HW_8/MNIST4.csv');
data = data{:,:};
[num_rows, num_cols] = size(data);

% Obtain mean vector
mean_vector = zeros(1, num_cols);
for i = 1 : num_rows
    mean_vector = mean_vector + data(i,:);
end
mean_vector = (1 / num_rows) * mean_vector;

% Obtain Data Covariance Matrix S
S = (data - repmat(mean_vector,[400,1]))'*(data - repmat(mean_vector,[400,1]));

% Get eigenvalues and eigenvectors of S
% Then sort so that largest eigenvalues come first
[V,D] = eig(S);
[d,ind] = sort(diag(D), 'descend');

Ds = D(ind,ind);
Vs = V(:,ind);

scatter(sort(ind(1:100) - 684, 'ascend'), d(1:100));
title("100 Largest Eigenvalues");
xlabel("Index");
ylabel("Eigenvalue");


% Visualize first 4 eigenvectors
eig_1 = Vs(:,1).';
eig_2 = Vs(:,2).';
eig_3 = Vs(:,3).';
eig_4 = Vs(:,4).';

% Reshape the first image
im_1 = zeros(28);
im_1(1,:) = eig_1(1, 28);
for i = 1 : 27
    im_1(i+1 ,:) = eig_1(i*28 + 1:(i*28 + 28));
end
max_val = max(max(im_1));

imshow(im_1, [0,255]);

% Correct Code from John
X3_average = mean(X3);
S = (X3-repmat(X3_average,[400,1]))'*(X3-repmat(X3_average,[400,1]));
[V,E] = eig(S,'vector');
figure
plot(fliplr(E(784-99:784)'))
title('The largest 100 eigenvalues')
%Visualize the eigenvectors
figure
for i = 1:4
    subplot(1,4,i)
    imshow(reshape(V(:,785-i),[28,28]),[])
    display(max(V(:,785-i)))
    titlechar = sprintf('u%d',i);
    title(titlechar)
end