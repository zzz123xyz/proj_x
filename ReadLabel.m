function Y = ReadLabel(label_path, option)
%ReadLabel read labels in the text files
%   label_path : path of labels
%   option : choose test or train
    listing = dir(fullfile(label_path,['*',option,'*','.txt']));
    nItem = numel(listing);
    
    for i = 1:nItem
       fid = fopen([label_path,listing(i).name]);
       Y(:,i) = cell2mat(textscan(fid,'%d'));
       fclose(fid);
    end


