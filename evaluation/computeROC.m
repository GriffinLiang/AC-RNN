function roc = computeROC(scores, labels)
% From Derek
[sval, sind] = sort(scores, 'descend');
roc.tp = cumsum(labels(sind)==1);
roc.fp = cumsum(labels(sind)==0);
roc.conf = sval;

roc = computeROCArea(roc);
roc.p = roc.tp ./ (roc.tp + roc.fp);
roc.r = roc.tp / roc.tp(end);
