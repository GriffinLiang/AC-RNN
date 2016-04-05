function [W_hv, W_hh, W_oh, b_h, b_o, h0] = parameter_init_RNN(theta, RNN)
    start_idx = 1 ; end_idx = RNN.h*RNN.v;
    W_hv = reshape(theta(start_idx:end_idx), RNN.h, RNN.v);

    start_idx = end_idx+1; end_idx = end_idx+RNN.h*RNN.h;
    W_hh = reshape(theta(start_idx:end_idx), RNN.h, RNN.h);

    start_idx = end_idx+1; end_idx = end_idx+RNN.z*RNN.h;
    W_oh = reshape(theta(start_idx:end_idx), RNN.z, RNN.h);

    start_idx = end_idx+1; end_idx = end_idx+RNN.h;
    b_h = theta(start_idx:end_idx);

    start_idx = end_idx+1 ; end_idx = end_idx+RNN.z;
    b_o = theta(start_idx:end_idx);

    start_idx = end_idx+1 ; end_idx = end_idx+RNN.h;
    h0 = theta(start_idx:end_idx);
    
    if(end_idx ~= size(theta, 1))
        error('RNN Paramter Siz does not match');
    end
end