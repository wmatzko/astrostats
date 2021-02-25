function [contents,filename] = urlcopy(URL,fullpath,basedir)
%URLCOPY Copy the contents of a file at a URL to local file
%
%   URLCOPY('URL') Copies the contents of URL to a local file.
%
%   URLCOPY('URL',fullpath) if fullpath = 0, puts file in local directory.
%   If fullpath = 1, puts file in directory with name determined from URL.
%   For example, if URL = 'http://www.abc.com/file.dat', file.dat is placed
%   in directory ./www.abc.com.
%
%   URLCOPY('URL',fullpath,Basedir) puts file in directory Basedir if
%   fullpath = 0.  If fullpath = 1, file is placed in
%   [Basedir,filesep(),'www.abc.com'].
%
%   See also URLREAD, URLWRITE.

if (nargin < 2)
  fullpath = 0;
end
if (nargin < 3)
  basedir = pwd();
end

[directory,filename] = fileparts(URL);
contents = urlread(URL);

if (fullpath == 1)
  I = findstr('://',URL);
  if ~isempty(I)
    directory = [basedir,directory(I(1)+2:end)];
    if (exist(directory) == 0)
      mkdir(directory);
    end
    if (exist(directory) ~= 7)
      error(sprintf(['urlcopy: Cannot create directory (%s): non-directory '...
		    'with that name exists'],directory));
    end
  else
    error(sprintf('urlcopy: Given URL (%s) does not appear to be a valid URL',URL));
  end
  filename = [directory,filesep(),filename];
end

fid = fopen(filename,'wt');
fprintf(fid,contents);
fclose(fid);
