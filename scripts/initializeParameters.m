function theta = initializeParameters(hiddenSize, visibleSize)

%% Initialize parameters randomly based on layer sizes.
r  = sqrt(6) / sqrt(hiddenSize+visibleSize);   % we'll choose weights uniformly from the interval [-r, r]
W = rand(hiddenSize, visibleSize) * 2 * r - r;
theta = W(:);
end
