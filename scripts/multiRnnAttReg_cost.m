function [cost, grad] = multiRnnAttReg_cost(theta, att_set, data, sequence_label, ...
                                            RNN, lambda, weight)

global useGpu ;

if( useGpu == true)
    data = gpuArray( data );
end


n_data = size(data, 2);
n_label = size(att_set, 2);
assert(n_label == size(sequence_label, 1))

cost = 0 ;

[W_hv, W_hh, W_oh, b_h, b_o, h0] = parameter_init_RNN(theta, RNN);
dW_hv = zeros(size(W_hv)); dW_hh = zeros(size(W_hh)); dW_oh = zeros(size(W_oh));
db_h = zeros(size(b_h)); db_o = zeros(size(b_o)); dh0 = zeros(size(h0));

atn{1} = bsxfun(@rdivide, att_set, sum(att_set, 1));
% multip scale
atn{1} = atn{1}*size(att_set, 1);
u{1} = W_hv*atn{1} + W_hh*repmat(h0, 1, n_label) + repmat(b_h, 1, n_label);
h{1} = sigmoid(u{1});

for ii = 2:RNN.T
    M{ii-1} = W_hv'*h{ii-1} ;
    M{ii-1} = bsxfun(@minus, M{ii-1}, max(M{ii-1}, [], 1)) ;
    atn{ii} = bsxfun(@rdivide, exp(M{ii-1}), sum(exp(M{ii-1})))*size(att_set, 1);
    u{ii} = W_hv*atn{ii} + W_hh*h{ii-1} + repmat(b_h, 1, n_label);
    h{ii} = sigmoid(u{ii});
end

o = W_oh*h{RNN.T} + repmat(b_o, 1, n_label);
z = sigmoid(o'*data);

if (exist('weight', 'var'))
    cost = cost - (1/n_data)*sum(sum(weight.*(sequence_label.*log(z) +...
                  (1-sequence_label).*log(1 - z)))) ;
    dW_oh = dW_oh + (1/n_data)*data*(weight.*(z-sequence_label))'*h{RNN.T}' ;
    db_o = db_o + (1/n_data)*sum(data*(weight.*(z - sequence_label))', 2);  
    dh{RNN.T} = (1/n_data)*W_oh'*(data*(weight.*(z-sequence_label))');
else
    cost = cost - (1/n_data)*sum(sum(sequence_label.*log(z) + ...
                            (1-sequence_label).*log(1 - z))) ;
    dW_oh = dW_oh + (1/n_data)*data*(z - sequence_label)'*h{RNN.T}';
    db_o = db_o + (1/n_data)*sum(data*(z - sequence_label)', 2);
    dh{RNN.T} = (1/n_data)*W_oh'*(data*(z-sequence_label)');
end
        
for jj = RNN.T:-1:2
    dW_hv = dW_hv + dh{jj}.*h{jj}.*(1-h{jj})*atn{jj}';
    a = size(att_set, 1)*W_hv'*(dh{jj}.*h{jj}.*(1-h{jj}));
    dW_hv = dW_hv + h{jj-1}*(exp(M{jj-1}).*bsxfun(@minus, bsxfun(@times, 1./sum(exp(M{jj-1})), a), ...
                                sum(bsxfun(@rdivide, exp(M{jj-1}), sum(exp(M{jj-1})).^2).*a)))';
    
    dW_hh = dW_hh + dh{jj}.*h{jj}.*(1-h{jj})*h{jj-1}';
    db_h = db_h + sum(dh{jj}.*h{jj}.*(1-h{jj}), 2);
    dh{jj-1} = W_hh'*(dh{jj}.*h{jj}.*(1-h{jj}));
    dh{jj-1} = dh{jj-1} + W_hv*(exp(M{jj-1}).*bsxfun(@minus, bsxfun(@times, 1./sum(exp(M{jj-1})), a), ...
                                sum(bsxfun(@rdivide, exp(M{jj-1}), sum(exp(M{jj-1})).^2).*a))); 
end

dW_hv = dW_hv + dh{1}.*h{1}.*(1-h{1})*atn{1}';
dW_hh = dW_hh + dh{1}.*h{1}.*(1-h{1})*repmat(h0, 1, n_label)';
db_h = db_h + sum(dh{1}.*h{1}.*(1-h{1}), 2);
dh0 = dh0 + sum(W_hh'*(dh{1}.*h{1}.*(1-h{1})), 2);
clear h dh o z


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
