warning off
close all
clear
clc
data = readmatrix('data.csv');
w = 1;
s = 48;
m = 4000;
n = 200;
input_train = [];
for i = 1:m
    xx = data(1+w*(i-1):w*(i-1)+s,:);
    xx = xx(:);
    input_train = [input_train, xx];
end
output_train = data(2:m+1, 1)';
input_test = [];
for i = m+1:m+n
    xx = data(1+w*(i-1):w*(i-1)+s,:);
    xx = xx(:);
    input_test = [input_test, xx];
end
output_test = data(m+2:m+n+1, 1)';

[inputn, inputps] = mapminmax(input_train, 0, 1);
[outputn, outputps] = mapminmax(output_train);
inputn_test = mapminmax('apply', input_test, inputps);

inputn = double(reshape(inputn, [size(inputn, 1), 1, 1, size(inputn, 2)]));
inputn_test = double(reshape(inputn_test, [size(inputn_test, 1), 1, 1, size(inputn_test, 2)]));
outputn = double(outputn)';

layers = [
    imageInputLayer([size(inputn, 1), 1, 1], 'Name', 'input')
    convolution2dLayer([3, 1], 16, 'Stride', [1, 1], 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'BN1')
    reluLayer('Name', 'relu1')
    convolution2dLayer([3, 1], 32, 'Stride', [1, 1], 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'BN2')
    reluLayer('Name', 'relu2')
    dropoutLayer(0.2, 'Name', 'dropout')
    fullyConnectedLayer(1, 'Name', 'fc')
    regressionLayer('Name', 'output')
];

options = trainingOptions('adam', ...
    'MaxEpochs', 200, ...
    'InitialLearnRate', 0.005, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 300, ...
    'L2Regularization', 1e-2, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', false);

net = trainNetwork(inputn, outputn, layers, options);
output_sim_train = predict(net, inputn);
output_sim_test = predict(net, inputn_test);
output_sim_train = mapminmax('reverse', output_sim_train, outputps);
output_sim_test = mapminmax('reverse', output_sim_test, outputps);
mse_train = mean((output_train' - output_sim_train).^2);
mse_test = mean((output_test' - output_sim_test).^2);

% rmse_train = sqrt(mse_train);
% rmse_test = sqrt(mse_test);

figure
plot(1:length(output_test), output_test, 'r-', 1:length(output_test), output_sim_test, 'b-', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
title(['CNN: 测试集预测结果对比 MSE=', num2str(mse_test)])
xlim([1, length(output_test)])
grid on

R2_train = 1 - sum((output_train' - output_sim_train).^2) / sum((output_train' - mean(output_train')).^2);
R2_test = 1 - sum((output_test' - output_sim_test).^2) / sum((output_test' - mean(output_test')).^2);

mae_train = mean(abs(output_sim_train - output_train'));
mae_test = mean(abs(output_sim_test - output_test'));

mbe_train = mean(output_sim_train - output_train');
mbe_test = mean(output_sim_test - output_test');

mape_train = mean(abs((output_sim_train - output_train') ./ output_train'));
mape_test = mean(abs((output_sim_test - output_test') ./ output_test'));

fprintf('CNN - 训练集 MSE = %.4f\n', mse_train);
fprintf('CNN - 测试集 MSE = %.4f\n', mse_test);
fprintf('CNN - 训练集 RMSE = %.4f\n', rmse_train);
fprintf('CNN - 测试集 RMSE = %.4f\n', rmse_test);
fprintf('CNN - 训练集 R2 = %.4f\n', R2_train);
fprintf('CNN - 测试集 R2 = %.4f\n', R2_test);
fprintf('CNN - 训练集 MAE = %.4f\n', mae_train);
fprintf('CNN - 测试集 MAE = %.4f\n', mae_test);
fprintf('CNN - 训练集 MBE = %.4f\n', mbe_train);
fprintf('CNN - 测试集 MBE = %.4f\n', mbe_test);
fprintf('CNN - 训练集 MAPE = %.4f\n', mape_train);
fprintf('CNN - 测试集 MAPE = %.4f\n', mape_test);
