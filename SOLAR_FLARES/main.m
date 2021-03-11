clear;
Year_o = 2000;
Year_f = 2003;

if ~exist('xray.mat')
  prep; % Get data, write xray.mat
end

compute; % Analysis of times in xray.mat
