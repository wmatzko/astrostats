% Data source:
% ftp://ftp.ngdc.noaa.gov/STP/SOLAR_DATA/SOLAR_FLARES/XRAY_FLARES/
% See ftp://ftp.ngdc.noaa.gov/STP/SOLAR_DATA/SOLAR_FLARES/XRAY_FLARES/xray.fmt.REVISED
% for information.

% This program extracts the start times of flares listed in these files
% (as determined from columns 14-17).
% The matrix Time_Start contains columns of [Year, Month, Day, Hour, Minute]

URLbase = 'ftp://ftp.ngdc.noaa.gov/STP/SOLAR_DATA/SOLAR_FLARES/XRAY_FLARES/';
Time_Start = [];
for year = Year_o:Year_f

%  i = year-2000+1;

  i = 1;
  clear YMD HM1;
  file_name = sprintf('xray%d',year);
  URL = [URLbase,file_name];

  if (~exist(file_name))
    fprintf('prep: downloading %s\n',URL);
    urlcopy(URL);
  else
    fprintf('prep: found %s on disk.  No download performed.\n',URL);
  end
  
  fid = fopen(file_name,'r');
  line = '';
  endoffile = 0;
  % This while loop could be easily vectorized if files had a uniform number
  % of columns.  However, the number of spaces at the end of a file is
  % not always uniform (as, for example, in xray2000).
  while (endoffile == 0)
    line = fgetl(fid);
    % length(line) > 1 takes care of case when ^M is returned at end of file
    % (as in the file xray2003) and case when line == -1 (which means EOF).
    if (length(line) > 1)
      YMD(i,:) = [year,str2num(line(8:9)),str2num(line(10:11))];
      HM1(i,:) = [str2num(line(14:15)),str2num(line(16:17))];
      i = i+1;
    else
      endoffile = 1;
    end
  end
  fclose(fid);
i  
  tmp = [YMD,HM1];
  Time_Start = [Time_Start ; tmp];
  
end

save -V6 xray.mat Time_Start 
save -ascii xray.txt Time_Start
