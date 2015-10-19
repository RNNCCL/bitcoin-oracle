
function [X,y] = parseData2(data)

t = cputime;

[m n] = size(data);

X = data(:,1);
y = data(:,2);
end

