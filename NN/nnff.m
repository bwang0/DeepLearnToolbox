function nn = nnff(nn, x, y)
%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)

    n = nn.n;
    m = size(x, 1);
    
    x = [ones(m,1) x];
    nn.a{1} = x;
    
    for i = 2 : n
      nn.a{i} = zeros(m,nn.size(i));
    end

    % weights format
    % 1: input layer. 4: output layer. 2, 3 hidden layers.
    %   1    2    3    4
    % 1 0    0    0    0
    % 2 1->2 0    0    0
    % 3 1->3 2->3 0    0
    % 4 1->4 2->4 3->4 0
    for j = 1 : n-1
      for i = j+1 : n
        if numel(nn.W{i,j}) == 0 % no connection j->i
          continue;
        end
        
        nn.a{i} = nn.a{i} + nn.a{j}*nn.W{i,j}';
      end
      
      % we are guaranteed now that the j+1 layer would have received
      % contributions from all j layer and below.
      % not for last layer, thus j+1 < n is required
      if i == n && j+1 < n
        switch nn.activation_function
          case 'sigm'
            nn.a{j+1} = sigm(nn.a{j+1});
          case 'tanh_opt'
            nn.a{j+1} = tanh_opt(nn.a{j+1});
        end
        
        %dropout
        if(nn.dropoutFraction > 0)
            if(nn.testing)
                nn.a{j+1} = nn.a{j+1}.*(1 - nn.dropoutFraction);
            else
                nn.dropOutMask{j+1} = (rand(size(nn.a{j+1}))>nn.dropoutFraction);
                nn.a{j+1} = nn.a{j+1}.*nn.dropOutMask{j+1};
            end
        end
        
        %calculate running exponential activations for use with sparsity
        if(nn.nonSparsityPenalty>0)
            nn.p{j+1} = 0.99 * nn.p{j+1} + 0.01 * mean(nn.a{j+1}, 1);
        end
        
        %Add the bias term
        nn.a{j+1} = [ones(m,1) nn.a{j+1}];
      end  
    end
    switch nn.output 
      case 'sigm'
        nn.a{n} = sigm(nn.a{n});
      case 'linear'
        % Nothing needed to be done
        % nn.a{n} = nn.a{n}; 
      case 'softmax'
        nn.a{n} = exp(bsxfun(@minus, nn.a{n}, max(nn.a{n},[],2)));
        nn.a{n} = bsxfun(@rdivide, nn.a{n}, sum(nn.a{n}, 2));
    end
    
%     %feedforward pass
%     for i = 2 : n-1
%         switch nn.activation_function 
%             case 'sigm'
%                 % Calculate the unit's outputs (including the bias term)
%                 nn.a{i} = sigm(nn.a{i - 1} * nn.W{i - 1}');
%             case 'tanh_opt'
%                 nn.a{i} = tanh_opt(nn.a{i - 1} * nn.W{i - 1}');
%         end
%         
%         %dropout
%         if(nn.dropoutFraction > 0)
%             if(nn.testing)
%                 nn.a{i} = nn.a{i}.*(1 - nn.dropoutFraction);
%             else
%                 nn.dropOutMask{i} = (rand(size(nn.a{i}))>nn.dropoutFraction);
%                 nn.a{i} = nn.a{i}.*nn.dropOutMask{i};
%             end
%         end
%         
%         %calculate running exponential activations for use with sparsity
%         if(nn.nonSparsityPenalty>0)
%             nn.p{i} = 0.99 * nn.p{i} + 0.01 * mean(nn.a{i}, 1);
%         end
%         
%         %Add the bias term
%         nn.a{i} = [ones(m,1) nn.a{i}];
%     end
%     switch nn.output 
%         case 'sigm'
%             nn.a{n} = sigm(nn.a{n - 1} * nn.W{n - 1}');
%         case 'linear'
%             nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
%         case 'softmax'
%             nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
%             nn.a{n} = exp(bsxfun(@minus, nn.a{n}, max(nn.a{n},[],2)));
%             nn.a{n} = bsxfun(@rdivide, nn.a{n}, sum(nn.a{n}, 2)); 
%     end

    %error and loss
    nn.e = y - nn.a{n};
    
    switch nn.output
        case {'sigm', 'linear'}
            nn.L = 1/2 * sum(sum(nn.e .^ 2)) / m; 
        case 'softmax'
            nn.L = -sum(sum(y .* log(nn.a{n}))) / m;
    end
end
