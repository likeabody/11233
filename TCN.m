clear
close all
clc

% 读取数据
data =  readmatrix('data.csv');
data = data(:,1:6);

w = 1;            % 滑动窗口大小
s = 48;           % 前24小时数据
m = 4000;         % 训练集样本数
n = 200;          % 测试集样本数

% 训练集输入输出
input_train = [];
for i = 1:m
    xx = data(1+w*(i-1):w*(i-1)+s,:);
    xx = xx(:);
    input_train = [input_train, xx];
end
output_train = data(2:m+1,1)';

% 测试集输入输出
input_test = [];
for i = m+1:m+n
    xx = data(1+w*(i-1):w*(i-1)+s,:);
    xx = xx(:);
    input_test = [input_test, xx];
end
output_test = data(m+2:m+n+1,1)';

%% 数据归一化
[inputn, inputps] = mapminmax(input_train, 0, 1);
[outputn, outputps] = mapminmax(output_train);
inputn_test = mapminmax('apply', input_test, inputps);

%% TCN网络结构参数
numFeatures = size(input_test,1);
outputSize = 1;
numFilters = 64;
filterSize = 5;
dropoutFactor = 0.2;
numBlocks = 6;

% TCN网络结构
layer = sequenceInputLayer(numFeatures, Normalization="rescale-symmetric", Name="input");
lgraph = layerGraph(layer);
outputName = layer.Name;

for i = 1:numBlocks
    dilationFactor = 2^(i-1);

    layers = [
        convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal",Name="conv1_"+i)
        layerNormalizationLayer
        dropoutLayer(dropoutFactor)
        convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal")
        layerNormalizationLayer
        reluLayer
        dropoutLayer(dropoutFactor)
        additionLayer(2,Name="add_"+i)];
    lgraph = addLayers(lgraph,layers);
    lgraph = connectLayers(lgraph,outputName,"conv1_"+i);

    % Skip connection
    if i == 1
        layer = convolution1dLayer(1,numFilters,Name="convSkip");
        lgraph = addLayers(lgraph,layer);
        lgraph = connectLayers(lgraph,outputName,"convSkip");
        lgraph = connectLayers(lgraph,"convSkip","add_" + i + "/in2");
    else
        lgraph = connectLayers(lgraph,outputName,"add_" + i + "/in2");
    end

    outputName = "add_" + i;
end

layers = [
    fullyConnectedLayer(outputSize,Name="fc")
    tanhLayer('Name','tanh')
    regressionLayer('Name','output')];
lgraph = addLayers(lgraph,layers);
lgraph = connectLayers(lgraph,outputName,"fc");

%% 训练选项
options = trainingOptions('adam', ...
    'MaxEpochs', 500, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.0005, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 300, ...
    'LearnRateDropFactor', 0.001, ...
    'L2Regularization', 0.001, ...
    'ExecutionEnvironment', 'gpu', ...
    'Verbose', 1, ...
    'Plots', 'training-progress');

%% 网络训练
net = trainNetwork(inputn, outputn, lgraph, options);

%% 仿真预测与反归一化
an = net.predict(inputn_test);
test_simu = mapminmax('reverse', an, outputps);

% 误差指标
error0 = output_test - test_simu;
mse0 = mse(output_test, test_simu);

%% 绘图
figure
plot(output_test, 'b-', 'markerfacecolor', [0.5,0.5,0.9], 'MarkerSize', 6)
hold on
plot(test_simu, 'r--', 'MarkerSize', 6)
title(['TCN的mse误差：', num2str(mse0)])
legend('真实y', '预测的y')
xlabel('样本数')
ylabel('负荷值')
box off
set(gcf,'color','w')

%% 评价指标
ae = abs(test_simu - output_test);
rmse = (mean(ae.^2)).^0.5;
mse = mean(ae.^2);
mae = mean(ae);
mape = mean(ae ./ test_simu);
[R, r] = corr(output_test, test_simu);
R2 = 1 - norm(output_test - test_simu)^2 / norm(output_test - mean(output_test))^2;
disp('预测结果评价指标：')
disp(['RMSE = ', num2str(rmse)])
disp(['MSE  = ', num2str(mse)])
disp(['MAE  = ', num2str(mae)])
disp(['MAPE = ', num2str(mape)])
disp(['决定系数R^2为：  ', num2str(R2)])