clear;clc;
addpath D:\Dataset\Attribute\lfwa
load('lfw_att_40','label') ;
load('LFWA_VGG_Face_center');
attribute_label = label' ;

data = bsxfun(@rdivide, feaTrain, sqrt(sum(feaTrain.^2))) ;
nData = size(data, 2);

train_data = data(:, mod(1:nData, 10)<6);
val_data = data(:, mod(1:nData, 10)==6);
test_data = data(:, mod(1:nData, 10)>6);

train_attribute_labels = attribute_label(:, mod(1:nData, 10)<6);
val_attribute_labels = attribute_label(:, mod(1:nData, 10)==6);
test_attribute_labels = attribute_label(:, mod(1:nData, 10)>6);

clear feaTrain data attribute_label label 

fid = 1;
%% Double Attribute Learning 
DouAtt_matrix = [];
for ii = 1:size(train_attribute_labels, 1)
    for jj = ii+1:size(train_attribute_labels, 1)
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

lambda = 10.^(-2);
h_size = 60;
v_size = size(train_attribute_labels, 1);
fprintf(fid, 'Double Attribute lambda:%f, h_size:%d\n', lambda, h_size);
z_size = size(train_data, 1);
n_att{1} = size(train_attribute_labels, 1);
T = 2;
attEmbed{1}{1} = 32*eye(size(train_attribute_labels, 1));
attEmbed{2}{1} = attEmbed{1}{1}(:, DouAtt_matrix(:,1));
attEmbed{2}{2} = attEmbed{1}{1}(:, DouAtt_matrix(:,2));
W_hv = initializeParameters(h_size,v_size);
W_hh = initializeParameters(h_size,h_size);
W_oh = initializeParameters(z_size,h_size);
b_h = initializeParameters(h_size, 1);
b_o = initializeParameters(z_size, 1);
h0 = initializeParameters(h_size, 1);
theta = [W_hv(:); W_hh(:); W_oh(:); b_h(:); b_o(:); h0(:)];
RNN.v = v_size; RNN.h = h_size; RNN.z = z_size; RNN.T = T;
sequence_label{1} = train_attribute_labels;
sequence_label{2} = tr_dou_att_labels;
options.maxIter = 400 ;
options.Method = 'L-BFGS'; 
options.display = 'on';        
[OptTheta, cost] = minFunc( @(p) multiRnnReg_cost(p, attEmbed, train_data, ...
                          sequence_label, RNN, lambda), theta, options);    
[W_hv, W_hh, W_oh, b_h, b_o, h0] = parameter_init_RNN(OptTheta, RNN);
clear u h o

sequence_label_val{1} = val_attribute_labels;
sequence_label_val{2} = val_dou_att_labels;
sequence_label_test{1} = test_attribute_labels;
sequence_label_test{2} = te_dou_att_labels;
attEmbed{2}{1} = attEmbed{1}{1}(:, DouAtt_matrix(:,1));
attEmbed{2}{2} = attEmbed{1}{1}(:, DouAtt_matrix(:,2));
n_att{T} = size(attEmbed{T}{1}, 2) ;
u{1} = W_hv*attEmbed{T}{1} + W_hh*repmat(h0, 1, n_att{T}) + repmat(b_h, 1, n_att{T});
h{1} = sigmoid(u{1});       
for jj = 2:T
    u{jj} = W_hv*attEmbed{T}{jj} + W_hh*h{jj-1} + repmat(b_h, 1, n_att{T});
    h{jj} = sigmoid(u{jj});
end
o = W_oh*h{T} + repmat(b_o, 1, n_att{T});
predProbVal{T} = sigmoid(o'*val_data);
predProbTest{T} = sigmoid(o'*test_data);
auc_val{T} = computeAUC(predProbVal{T}, sequence_label_val{T}) ;
auc_test{T} = computeAUC(predProbTest{T}, sequence_label_test{T}) ;
fprintf(fid, 'AUC Val:%0.4f\tTest:%0.4f\n', mean(auc_val{T}), mean(auc_test{T}));    
     