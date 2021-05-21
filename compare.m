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
% initialize phased error
error = zeros(T ,1);
for i=1:T
    % model training
    model = fitensemble(X_train,y_train,'AdaBoostM1',i,'Tree');
    % predict with model
    predicted_label = predict(model, X_test);
    % store error of model
    error(i) = (1 - sum(predicted_label == y_test) / (m - 300)) * 100;
end
% get the training error
rsLoss = resubLoss(model,'Mode','Cumulative');
% draw graph
plot(rsLoss);
hold on;
plot(error);
% set graph's information
title('Train Error and Test Error against Boosting Time')
xlabel('Number of Learning Cycles');
ylabel('Loss %');
legend({'train error','test error'})

