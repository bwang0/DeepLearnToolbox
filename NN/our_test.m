load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

% normalize
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

%%
rand('state',0);
simple_arch_mask = [0 0 0 0; 1 0 0 0; 0 1 0 0; 0 0 1 0];
full_arch_mask = [0 0 0 0; 1 0 0 0; 1 1 0 0; 1 1 1 0];
nn = nnsetup([784 300 100 10],full_arch_mask);

nn.activation_function = 'sigm';    %  Sigmoid activation function
nn.weightPenaltyL2 = 1e-4;  %  L2 weight decay
opts.numepochs =  1;        %  Number of full sweeps through data
nn.learningRate = 1;                %  Sigm require a lower learning rate
opts.numepochs =  1;                %  Number of full sweeps through data
opts.batchsize = 100;               %  Take a mean gradient step over this many samples

for i = 1:100
  nn = nntrain(nn, train_x, train_y, opts);
  [er, bad] = nntest(nn, test_x, test_y);
  er
end

