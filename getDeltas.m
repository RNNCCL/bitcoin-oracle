function delta = getDeltas(k)
	% clone an shift
	k2 = [0 k];
	k2(:,end) = [];

	% get the difference
	delta = k - k2;
	delta(:,1) = [];
end
