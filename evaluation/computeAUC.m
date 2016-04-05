function auc = computeAUC(scores, labels)

if(size(scores, 1) ~= size(labels, 1) || size(scores, 2) ~= size(labels, 2))
    error('Input dimension does not match!')
end

nAtt = size(labels, 1) ;
auc = zeros(1, nAtt) ;

for iAtt = 1:nAtt 
    if(size(unique(labels(iAtt,:)), 2) == 1)
        continue
    end
    
    roc = computeROC(scores(iAtt,:)', labels(iAtt,:)') ;       
	auc(iAtt) = roc.area ;   
end

