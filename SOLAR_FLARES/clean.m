%delete('xray.mat');
delete('*~');
for year = Year_o:Year_f
  delete(sprintf('xray%d',year));
end
