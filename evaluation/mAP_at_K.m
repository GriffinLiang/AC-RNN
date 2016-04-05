function score = mAP_at_K(label, prediction, k)
%MEANAVERAGEPRECISIONATK   Calculates the average precision at k
%   score = meanAveragePrecisionAtK(actual, prediction, k)
%
%   actual is a cell array of vectors
%   prediction is a cell array of vectors
%   k is an integer
%
%   Author: Kongming Liang
assert(sum(size(label) == size(prediction))==2)
n_att = size(label, 1);

for ii = 1:n_att
    actual{ii} = find(label(ii,:)==1);
    [~, temp] = sort(prediction(ii,:), 'descend');
    order{ii} = temp;
end

score = meanAveragePrecisionAtK(actual, order, k);