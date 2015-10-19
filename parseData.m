% parseData(priCols) convert bitstampUSD.csv to matrix X (consecutive priCols prices and amounts) 
% and vector y (the price just after those consecutive prices)

function [X,y] = parseData(data, priCols)

t = cputime;

[m n] = size(data);



X = zeros(1,priCols*2);
y = zeros(1);
for i=1:m-priCols,
	% add prices
	newExample = data(i:i+priCols-1,2)';
	% add ammunts
	newExample = [newExample data(i:i+priCols-1,3)'];

	X = [X;newExample];
	y = [y;data(i+priCols,2)];
end
X(1,:) = [];
y(1,:) = [];
printf('Data parsed in in %f seconds\n',cputime-t);


end

