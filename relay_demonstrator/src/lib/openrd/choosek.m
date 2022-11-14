function M=choosek(n,k)
%M=choosek(n,k)

if k == 0
	M = zeros(1,n);
else
	M = [];
	for i = 0:n-k
		t = choosek(n-i-1,k-1);
		M = [M;zeros(size(t,1),i),ones(size(t,1),1),t];
	end
end

M = (M == 1);

