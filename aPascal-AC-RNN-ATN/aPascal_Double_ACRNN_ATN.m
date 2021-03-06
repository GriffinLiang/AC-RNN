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

fid = 1;

%% Double Attribute Learning 
DouAtt_matrix = [];
for ii = 1:64
    for jj = ii+1:64
        idx_ii = train_attribute_labels(ii,:);
        idx_jj = train_attribute_labels(jj,:);
        idx_ii_val = val_attribute_labels(ii,:);
        idx_jj_val = val_attribute_labels(jj,:);
        idx_ii_te = test_attribute_labels(ii,:);
        idx_jj_te = test_attribute_labels(jj,:);        
        if(sum(idx_ii & idx_jj) > 0 && sum(idx_ii_val & idx_jj_val) > 0 && ...
                sum(idx_ii_te & idx_jj_te) > 0)
            DouAtt_matrix = [DouAtt_matrix; ii jj];
        end
    end
end

tr_dou_att_labels = [];
val_dou_att_labels = [];
te_dou_att_labels = [];
for ii = 1:size(DouAtt_matrix, 1)
    att1 = DouAtt_matrix(ii, 1);
    att2 = DouAtt_matrix(ii, 2);
    tr_dou_att_labels = [tr_dou_att_labels; train_attribute_labels(att1,:) & ...
                                            train_attribute_labels(att2,:)];
    val_dou_att_labels = [val_dou_att_labels; val_attribute_labels(att1,:) & ...
                                            val_attribute_labels(att2,:)];
	te_dou_att_labels = [te_dou_att_labels; test_attribute_labels(att1,:) & ...
                                            test_attribute_labels(att2,:)];
end

att_set = zeros(64, size(DouAtt_matrix, 1));
for ii = 1:2
att_set(sub2ind(size(att_set), DouAtt_matrix(:,ii)', 1:size(DouAtt_matrix, 1))) = 1;
end

T = 2;
lambda = 10.^(-2);            
h_size = 60;
v_size = 64;
fprintf(fid, 'Double Attribute T:%d, lambda:%f, h_size:%d\n', T, lambda, h_size);
z_size = size(train_data, 1);
n_att = size(tr_dou_att_labels, 1);
W_hv = initializeParameters(h_size,v_size);
W_hh = initializeParameters(h_size,h_size);
W_oh = initializeParameters(z_size,h_size);
b_h = initializeParameters(h_size, 1);
b_o = initializeParameters(z_size, 1);
h0 = initializeParameters(h_size, 1);
OptTheta = [W_hv(:); W_hh(:); W_oh(:); b_h(:); b_o(:); h0(:)];
RNN.v = v_size; RNN.h = h_size; RNN.z = z_size; RNN.T = T;
sequence_label = tr_dou_att_labels;
options.maxIter = 400 ;
options.Method = 'L-BFGS'; 
options.display = 'on'; 
[OptTheta, cost] = minFunc( @(p) multiRnnAttReg_cost(p, att_set, train_data, ...
                sequence_label, RNN,lambda), OptTheta, options);    
[W_hv, W_hh, W_oh, b_h, b_o, h0] = parameter_init_RNN(OptTheta, RNN);

atn{1} = bsxfun(@rdivide, att_set, sum(att_set, 1));
% multip scale
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
auc_val = computeAUC(predProbVal, val_dou_att_labels) ;
fprintf(fid, 'Val mAUC:%0.4f\t', mean(auc_val)) ;
predProbTe = sigmoid(o'*test_data);
auc_te = computeAUC(predProbTe, te_dou_att_labels) ;
fprintf(fid, 'Test mAUC:%0.4f\n', mean(auc_te)) ; 
