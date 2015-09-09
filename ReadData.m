function X = ReadData(data_path1)
%ReadData: read the features in text files
%   Detailed explanation goes here

    expression = '\s';
    fid = fopen(data_path1,'r');
    if fid > 0
     
%    
    t = 0;
        while ~feof(fid)    
            t = t+1;
            tline = fgetl(fid);
            matchStr = regexp(tline, expression, 'split');
            matchStr = str2double(matchStr(1:end-1));
            %X(t,:) = matchStr;
            X(:,t) = matchStr';
        end    
    
    else
        error('can not open the file');
    end
    
    fclose(fid);



