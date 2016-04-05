function [cost, grad] = multiRnnReg_cost(theta, attEmbed, data, sequence_label, RNN, ...
                                    lambda, weight)

global useGpu ;

if( useGpu == true)
    data = gpuArray( data );
end


n_data = size(data, 2);

cost = 0 ;
[W_hv, W_hh, W_oh, b_h, b_o, h0] = parameter_init_RNN(theta, RNN);
dW_hv = zeros(size(W_hv)); dW_hh = zeros(size(W_hh)); dW_oh = zeros(size(W_oh));
db_h = zeros(size(b_h)); db_o = zeros(size(b_o)); dh0 = zeros(size(h0));

for ii = RNN.T
    n_att = size(attEmbed{ii}{1}, 2) ;
    u{1} = W_hv*attEmbed{ii}{1} + W_hh*repmat(h0, 1, n_att) + repmat(b_h, 1, n_att);
    h{1} = sigmoid(u{1});        
    for jj = 2:ii
        u{jj} = W_hv*attEmbed{ii}{jj} + W_hh*h{jj-1} + repmat(b_h, 1, n_att);
        h{jj} = sigmoid(u{jj});
    end

	o = W_oh*h{ii} + repmat(b_o, 1, n_att);
    z = sigmoid(o'*data);
    
    if (exist('weight', 'var'))
        cost = cost - (1/n_data)*sum(sum(weight{ii}.*(sequence_label{ii}.*log(z) +...
                      (1-sequence_label{ii}).*log(1 - z)))) ;
        dW_oh = dW_oh + (1/n_data)*data*(weight{ii}.*(z-sequence_label{ii}))'*h{ii}' ;
        db_o = db_o + (1/n_data)*sum(data*(weight{ii}.*(z - sequence_label{ii}))', 2);  
        dh{ii} = (1/n_data)*W_oh'*(data*(weight{ii}.*(z-sequence_label{ii}))');
    else
        cost = cost - (1/n_data)*sum(sum(sequence_label{ii}.*log(z) + ...
                                (1-sequence_label{ii}).*log(1 - z))) ;
        dW_oh = dW_oh + (1/n_data)*data*(z - sequence_label{ii})'*h{ii}';
        db_o = db_o + (1/n_data)*sum(data*(z - sequence_label{ii})', 2);
        dh{ii} = (1/n_data)*W_oh'*(data*(z-sequence_label{ii})');
    end
    
    
 
    for jj = ii:-1:2
        dW_hv = dW_hv + dh{jj}.*h{jj}.*(1-h{jj})*attEmbed{ii}{jj}';
        dW_hh = dW_hh + dh{jj}.*h{jj}.*(1-h{jj})*h{jj-1}';
        db_h = db_h + sum(dh{jj}.*h{jj}.*(1-h{jj}), 2);
        dh{jj-1} = W_hh'*(dh{jj}.*h{jj}.*(1-h{jj}));
    end

    dW_hv = dW_hv + dh{1}.*h{1}.*(1-h{1})*attEmbed{ii}{1}';
    dW_hh = dW_hh + dh{1}.*h{1}.*(1-h{1})*repmat(h0, 1, n_att)';
    db_h = db_h + sum(dh{1}.*h{1}.*(1-h{1}), 2);
    dh0 = dh0 + sum(W_hh'*(dh{1}.*h{1}.*(1-h{1})), 2);
    clear h dh o z
end

cost = cost + 0.5*lambda*sum(theta.^2);
% W_hv, W_hh, W_oh, b_h, b_o, h0
dW_hv = dW_hv + lambda*W_hv; 
dW_hh = dW_hh + lambda*W_hh;
dW_oh = dW_oh + lambda*W_oh;
db_h = db_h + lambda*b_h;
db_o = db_o + lambda*b_o; 
dh0 = dh0 + lambda*h0;
grad = [dW_hv(:); dW_hh(:); dW_oh(:); db_h(:); db_o(:); dh0(:)];

if( useGpu == true)
    cost = gather(cost) ;
    grad = gather(grad) ;
end

end

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end