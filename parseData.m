% parseData(priCols) converts data matrix to matrix X (consecutive priCols prices and data) 
% and vector y (the price to predict just after those consecutive prices and data)

function [X,y] = parseData(data, priCols)

% t = cputime;

% data = flipud(data);

[m n] = size(data);


X = [];%zeros(1,priCols*6+(priCols-1)*6);
y = [];%zeros(1);
for i=1:m-priCols,
	ex = [];

	openPrice = data(i:i+priCols-1,1)';
	ex = [ex openPrice];
	deltaOpenPrice = getDeltas(openPrice);
	ex = [ex deltaOpenPrice];

	highPrice = data(i:i+priCols-1,2)';
	% ex = [ex highPrice];
	deltaHighPrice = getDeltas(highPrice);
	ex = [ex deltaHighPrice];


	lowPrice = data(i:i+priCols-1,3)';
	ex = [ex lowPrice];
	deltaLowPrice = getDeltas(lowPrice);
	ex = [ex deltaLowPrice];

	closePrice = data(i:i+priCols-1,4)';
	ex = [ex closePrice];
	deltaClosePrice = getDeltas(closePrice);
	ex = [ex deltaClosePrice];

	volumeBtc = data(i:i+priCols-1,5)';
	% ex = [ex volumeBtc];
	deltaVolumeBtc = getDeltas(volumeBtc);
	ex = [ex deltaVolumeBtc];

	volumeUsd = data(i:i+priCols-1,6)';
	% ex = [ex volumeUsd];
	deltaVolumeUsd = getDeltas(volumeUsd);
	% ex = [ex deltaVolumeUsd];

	weightedPrice = data(i:i+priCols-1,7)';
	% ex = [ex weightedPrice];
	deltaWeightedPrice = getDeltas(weightedPrice);
	% ex = [ex deltaWeightedPrice];

	

	X = [X;ex];

	% column to predict
	y = [y;data(i+priCols,4)];
end



% printf('Data parsed in in %f seconds\n',cputime-t);


end

