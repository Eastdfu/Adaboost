function [y_predict, acc, error] = test(X_test, y_test, inds, ss, xs, alphas, T)
% get amount of test samples
[m, ~] = size(X_test);
% initialize error and prediced label
y_predict = zeros(m, 1);
error = zeros(T ,1);
% for every step of learning cycle
for i = 1 : T
    % get the prediction result of weak learner
    y_temp = -1 * ones(m, 1);
    y_temp(X_test(:, inds(i)) > xs(i)) = 1;
    y_temp = y_temp * ss(i);
    % combine result of weak learners as a phased result
    y_predict = y_predict + alphas(i) * y_temp;
    yyy = sign(y_predict);
    % store phased loss of test
    error(i) = 1 - sum(yyy == y_test) / m;
end
error = error * 100;
% draw test error
plot(error);
% final accuracy
y_predict = sign(y_predict);
acc = sum(y_predict == y_test) / m;
end