clear; 
addpath D:\Dataset\Attribute\Imagenet\ ;
load('imagenet_attribute_25_BB_DeCAF.mat') ;
load('attrann.mat') ;

category_label = repmat(1:384, 25, 1) ;      
category_label = category_label(:) ;
attribute_label = attrann.labels' ;
attribute_label(attribute_label == 0) = 0.5 ;
attribute_label(attribute_label == -1) = 0 ;

data = bsxfun(@rdivide, feaTrain, sqrt(sum(feaTrain.^2))) ;
nData = size(data, 2);

train_data = data(:, mod(1:nData, 10)<6);
val_data = data(:, mod(1:nData, 10)==6);
test_data = data(:, mod(1:nData, 10)>6);

train_attribute_labels = attribute_label(:, mod(1:nData, 10)<6);
val_attribute_labels = attribute_label(:, mod(1:nData, 10)==6);
test_attribute_labels = attribute_label(:, mod(1:nData, 10)>6);

clear  data attribute_label category_label attrann feaTrain
%% Single attribute query
fid = 1;

att_set = eye(size(train_attribute_labels, 1));
T = 2;
lambda = 10.^(-3);            
h_size = 80;
v_size = size(train_attribute_labels, 1);
fprintf(fid, 'Single Attribute T:%d, lambda:%f, h_size:%d\n', T, lambda, h_size);
z_size = size(train_data, 1);
n_att = size(train_attribute_labels, 1);
W_hv = initializeParameters(h_size,v_size);
W_hh = initializeParameters(h_size,h_size);
W_oh = initializeParameters(z_size,h_size);
b_h = initializeParameters(h_size, 1);
b_o = initializeParameters(z_size, 1);
h0 = initializeParameters(h_size, 1);
OptTheta = [W_hv(:); W_hh(:); W_oh(:); b_h(:); b_o(:); h0(:)];
RNN.v = v_size; RNN.h = h_size; RNN.z = z_size; RNN.T = T;
sequence_label = train_attribute_labels;
weight = zeros(size(sequence_label));
for jj = 1:size(sequence_label, 1)
    pos_num = sum(sequence_label(jj, :) == 1);
    neg_num = sum(sequence_label(jj, :) == 0);
    weight(jj, sequence_label(jj, :)==1) = (pos_num + neg_num)/(2*pos_num);
    weight(jj, sequence_label(jj, :)==0) = (pos_num + neg_num)/(2*neg_num);
end
options.maxIter = 400 ;
options.Method = 'L-BFGS'; 
options.display = 'on'; 
[OptTheta, cost] = minFunc( @(p) multiRnnAttReg_cost(p, att_set, train_data, ...
                sequence_label, RNN,lambda,weight), OptTheta, options); 
[W_hv, W_hh, W_oh, b_h, b_o, h0] = parameter_init_RNN(OptTheta, RNN);

atn{1} = bsxfun(@rdivide, att_set, sum(att_set, 1));
atn{1} = atn{1}*size(att_set, 1);
u{1} = W_hv*atn{1} + W_hh*repmat(h0, 1, n_att) + repmat(b_h, 1, n_att);
h{1} = sigmoid(u{1});

for ii = 2:RNN.T
    M{ii-1} = W_hv'*h{ii-1} ;
    M{ii-1} = bsxfun(@minus, M{ii-1}, max(M{ii-1}, [], 1)) ;
    atn{ii} = bsxfun(@rdivide, exp(M{ii-1}), sum(exp(M{ii-1})));
    atn{ii} = atn{ii}*size(att_set, 1);
    u{ii} = W_hv*atn{ii} + W_hh*h{ii-1} + repmat(b_h, 1, n_att);
    h{ii} = sigmoid(u{ii});
end

o = W_oh*h{RNN.T} + repmat(b_o, 1, n_att);

predProbVal = sigmoid(o'*val_data);
auc_val = computeAUC(predProbVal, val_attribute_labels) ;
fprintf(fid, 'Val mAUC:%0.4f\t', mean(auc_val)) ; 
predProbTe = sigmoid(o'*test_data);
auc_te = computeAUC(predProbTe, test_attribute_labels) ;
fprintf(fid, 'Test mAUC:%0.4f\n', mean(auc_te)) ;    
