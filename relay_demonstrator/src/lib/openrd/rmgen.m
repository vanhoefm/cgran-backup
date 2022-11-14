function rm = rmgen(r,m)
%rm = rmgen(r,m)
%
%This function generates encoder and decoder structures for the Reed-
%Muller code RM(r,m). The output is a structure containing the following
%elements:
%  r - Reed-Muller code parameter
%  m - Reed-Muller code parameter
%  k - the code dimension
%  n - the code wordlength
%  G - generator matrix, of size k*n
%  mlv - majority-logic decoding information
%
%mlv is a size-(r+1) cell array, where element i contains the majority-logic
%rules for the rows of G consisting of i variables. v=mlv{1+i} is a three-
%dimensional array of dimensions (npar, vars, nbits), where nbits are the
%number of bits that v contains majority-logic rules for, npar is the number
%of parity checks to perform for each bit, and vars is the number of
%variables involved in each check. npar=2^(m-i), vars=2^i, and 
%nbits=nchoosek(m,i).

k = 0;
n = 2^m;

for ri = 0:r
	k = k + nchoosek(m,ri);
end

basevars = zeros(m,n);
for i = 1:m
	A = [ones(1,2^(m-i)), zeros(1,2^(m-i))];
	basevars(i,:) = repmat(A,1,n/length(A));
end

G = [];
for ri = 0:r
	vars = choosek(m,ri);
	for vi = 1:size(vars,1)
		row = prod([ones(1,n);basevars(vars(vi,:),:)],1);
		G = [G;row];
	end
end

mlv = cell(1,r+1);
for bi = 0:r
	varsets = choosek(m, m-bi);
	vecs = zeros(2^(m-bi), 2^bi, size(varsets, 1));
	for vi = 1:size(varsets, 1)
		varmat = basevars(varsets(vi,:),:);
		for ci = 0:2^(m-bi)-1
			ch = dec2bin(ci, m-bi)=='1';
			vars = ones(1,n);
			for chi = 1:m-bi
				if ch(chi) == 0
					vars = vars.*varmat(chi,:);
				else
					vars = vars.*(1-varmat(chi,:));
				end
			end
			vecs(1+ci,:,vi) = find(vars==1)-1;
		end
	end
	mlv{1+bi} = vecs;
end

rm.r = r;
rm.m = m;
rm.k = k;
rm.n = n;
rm.G = G;
rm.mlv = mlv;

