function [inds, ss, xs, alphas, H] = AB(X_train, y_train, T)
% get amount of test samples
[M, ~] = size(X_train);
% initialize distribution
D = 1 / M * ones(M, 1);
% initialize final result
H = zeros(M, 1);
% initialize stumps'information array
inds = zeros(1, T);
ss = zeros(1, T);
xs = zeros(1, T);
alphas = zeros(1, T);
% initialize phased error
train_error = zeros(1, T);
% for every step of learning cycle
for i = 1 : T
    % train a weak learner and store model with a weight alpha
    stump = build_stump(X_train, y_train, D);
    alpha = 1 / 2 * log((1 - stump.werr) / stump.werr);
    alphas(i) = alpha;
    inds(i) = stump.ind;
    ss(i) = stump.s;
    xs(i) = stump.x0;
    % predict via model
    y_temp = -1 * ones(M, 1);
    y_temp(X_train(:, stump.ind) > stump.x0) = 1;
    y_temp = y_temp * stump.s;
    % get the phased training error
    H = H + alpha * y_temp;
    train_error(i) = 100 * (1 - sum(sign(H) == y_train) / M);
    % update the distribution
    D = D .* exp(-alpha * y_train .* y_temp);
    D = D / sum(D);
end
% draw phased training error
plot(train_error, 'r');
hold on;
% final result and accuracy
H = sign(H);
H = sum(H == y_train) / M;
end