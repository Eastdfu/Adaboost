% preprocess data
A = importdata("wdbc_data.csv");
STR = A.textdata(:,2);
X = A.data;
STR = cell2mat(STR);
[m, n] = size(X);
y = ones(m, 1);
y(STR == 'B') = -1;
X_train = X(1:300,:);
y_train = y(1:300,:);
X_test = X(301:m,:);
y_test = y(301:m,:);
% define training times
T = 500;
% model training
[inds, ss, xs, alphas, acc_train] = AB(X_train, y_train, T);
% test model
[y_predict, acc_test, error] = test(X_test, y_test, inds, ss, xs, alphas, T);
% set graph's information
title('Train Error and Test Error against Boosting Time')
xlabel('Number of Learning Cycles');
ylabel('Loss %');
legend({'train error','test error'})
hold off;