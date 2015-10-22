
function [X,y] = parseData2(data)

[m n] = size(data);

X = data(:,1);
X = [X data(:,3)];

y = data(:,2);
end

