function nn = nnsetup(architecture,arch_mask)
%NNSETUP creates a Feedforward Backpropagate Neural Network
% nn = nnsetup(architecture) returns an neural network structure with n=numel(architecture)
% layers, architecture being a n x 1 vector of layer sizes e.g. [784 100 10]

    nn.size = architecture;
    nn.n = numel(nn.size);
    
    nn.activation_function              = 'tanh_opt';   %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
    nn.learningRate                     = 2;            %  learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
    nn.momentum                         = 0.5;          %  Momentum
    nn.scaling_learningRate             = 1;            %  Scaling factor for the learning rate (each epoch)
    nn.weightPenaltyL2                  = 0;            %  L2 regularization
    nn.nonSparsityPenalty               = 0;            %  Non sparsity penalty
    nn.sparsityTarget                   = 0.05;         %  Sparsity target
    nn.inputZeroMaskedFraction          = 0;            %  Used for Denoising AutoEncoders
    nn.dropoutFraction                  = 0;            %  Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
    nn.testing                          = 0;            %  Internal variable. nntest sets this to one.
    nn.output                           = 'sigm';       %  output unit 'sigm' (=logistic), 'softmax' and 'linear'
    
    % weights format
    % 1: input layer. 4: output layer. 2, 3 hidden layers.
    %   1    2    3    4
    % 1 0    0    0    0
    % 2 1->2 0    0    0
    % 3 1->3 2->3 0    0
    % 4 1->4 2->4 3->4 0
    nn.W = cell(nn.n);
    nn.vW = cell(nn.n);
    nn.p = cell(nn.n);
    
    for i = 2 : nn.n
      for j = 1 : i-1
        if arch_mask(i,j) == 0
          continue;
        end
        
        % weights and weight momentum
        nn.W{i,j} = (rand(nn.size(i), nn.size(j)+1) - 0.5) * 2 * 4 * sqrt(6 / (nn.size(i) + nn.size(j)));
        nn.vW{i,j} = zeros(size(nn.W{i,j}));

        % average activations (for use with sparsity)
        nn.p{i,j}     = zeros(1, nn.size(i));
      end
    end

end