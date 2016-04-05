clear; 
addpath D:\Dataset\Attribute\aPascal_aYahoo ;
load('aPascal_DeCAF.mat', 'aPascal_train', 'aPascal_test') ;
load('aPascal_Annotation.mat','apascal_train_attribute');
load('aPascal_Annotation.mat','apascal_test_attribute');

nData = size(aPascal_train, 2);
train_data = aPascal_train(:, mod(1:nData, 10) ~= 0);
val_data = aPascal_train(:, mod(1:nData, 10) == 0);
test_data = aPascal_test;
train_attribute_labels = apascal_train_attribute(:, mod(1:nData, 10) ~= 0);
val_attribute_labels = apascal_train_attribute(:, mod(1:nData, 10) == 0);
test_attribute_labels = apascal_test_attribute;

clear  aPascal_train apascal_train_category apascal_train_attribute
clear  aPascal_test apascal_test_category apascal_test_attribute

%% Single attribute query
fid = 1;

lambda = 10.^(-3);
h_size = 60;
v_size = 64;
fprintf(fid, 'Single Attribute lambda:%f, h_size:%d\n', lambda, h_size);
z_size = size(train_data, 1);
n_att = size(train_attribute_labels, 1);
T = 1;
attEmbed{1}{1} = 32*eye(n_att);
W_hv = initializeParameters(h_size,v_size);
W_hh = initializeParameters(h_size,h_size);
W_oh = initializeParameters(z_size,h_size);
b_h = initializeParameters(h_size, 1);
b_o = initializeParameters(z_size, 1);
h0 = initializeParameters(h_size, 1);
theta = [W_hv(:); W_hh(:); W_oh(:); b_h(:); b_o(:); h0(:)];
RNN.v = v_size; RNN.h = h_size; RNN.z = z_size; RNN.T = T;
sequence_label{1} = train_attribute_labels;
options.maxIter = 400 ;
options.Method = 'L-BFGS'; 
options.display = 'on';     
[OptTheta, cost] = minFunc( @(p) multiRnnReg_cost(p, attEmbed, train_data, ...
                          sequence_label, RNN, lambda), theta, options);    
[W_hv, W_hh, W_oh, b_h, b_o, h0] = parameter_init_RNN(OptTheta, RNN);
u{1} = W_hv*attEmbed{1}{1} + W_hh*repmat(h0, 1, n_att) + repmat(b_h, 1, n_att);
h{1} = sigmoid(u{1});
o{1} = W_oh*h{1} + repmat(b_o, 1, n_att);
predProbVal = sigmoid(o{1}'*val_data);
auc_val = computeAUC(predProbVal, val_attribute_labels) ;
fprintf(fid, 'Val mAUC:%0.4f\t', mean(auc_val)) ; 
predProbTe = sigmoid(o{1}'*test_data);
auc_te = computeAUC(predProbTe, test_attribute_labels) ;
fprintf(fid, 'Test mAUC:%0.4f\n', mean(auc_te)) ; 
fprintf(fid, '**************************************************\n') ;  