function nn = nnbp(nn)
%NNBP performs backpropagation
% nn = nnbp(nn) returns an neural network structure with updated weights 
    
    n = nn.n;
    sparsityError = 0;
    switch nn.output
        case 'sigm'
            d{n} = - nn.e .* (nn.a{n} .* (1 - nn.a{n}));
        case {'softmax','linear'}
            d{n} = - nn.e;
    end
    
    % weights format
    % 1: input layer. 4: output layer. 2, 3 hidden layers.
    %   1    2    3    4
    % 1 0    0    0    0
    % 2 1->2 0    0    0
    % 3 1->3 2->3 0    0
    % 4 1->4 2->4 3->4 0
    for j = (n - 1) : -1 : 2
        % Derivative of the activation function
        switch nn.activation_function
            case 'sigm'
                d_act = nn.a{j} .* (1 - nn.a{j});
            case 'tanh_opt'
                d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{j}.^2);
        end
        
        if(nn.nonSparsityPenalty>0)
            pi = repmat(nn.p{j}, size(nn.a{j}, 1), 1);
            sparsityError = [zeros(size(nn.a{j},1),1) nn.nonSparsityPenalty * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi))];
        end
        
        % Backpropagate first derivatives
        d{j} = sparsityError;
        for i = j+1 : n
          if numel(nn.W{i,j}) == 0
            continue;
          end
          
          if i == n % in this case in d{n} there is not the bias term to be removed
            d{j} = d{j} + d{i} * nn.W{i,j};
          else % in this case in d{i} the bias term has to be removed
            d{j} = d{j} + d{i}(:,2:end) * nn.W{i,j};
          end
        end        
        d{j} = d{j} .* d_act;
        
        % Dropout mask
        if(nn.dropoutFraction>0)
            d{j} = d{j} .* [ones(size(d{j},1),1) nn.dropOutMask{j}];
        end
    end

    nn.dW = cell(n);
    batch_size = size(d{n},1);
    for j = 1 : (n - 1)
      for i = j+1 : n
        if numel(nn.W) == 0
          continue;
        end
        
        if i==n
            nn.dW{i,j} = (d{i}' * nn.a{j}) / batch_size;
        else
            nn.dW{i,j} = (d{i}(:,2:end)' * nn.a{j}) / batch_size;
        end
      end
    end
end
