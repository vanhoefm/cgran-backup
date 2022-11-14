G=[1 0 1 1 0 1 1 1 0 0 0 1;
0 1 1 0 1 1 1 0 0 0 1 1;
1 1 0 1 1 1 0 0 0 1 0 1;
1 0 1 1 1 0 0 0 1 0 1 1;
0 1 1 1 0 0 0 1 0 1 1 1;
1 1 1 0 0 0 1 0 1 1 0 1;
1 1 0 0 0 1 0 1 1 0 1 1;
1 0 0 0 1 0 1 1 0 1 1 1;
0 0 0 1 0 1 1 0 1 1 1 1;
0 0 1 0 1 1 0 1 1 1 0 1;
0 1 0 1 1 0 1 1 1 0 0 1;
1 1 1 1 1 1 1 1 1 1 1 0];

pbits = zeros(1,4096);
numones = zeros(1,4096);

for k = 0:4095
	x = (dec2bin(k)=='1');
	x = [zeros(1,12-length(x)) x];
	y = mod(x*G, 2);
	v = sum(y.*(2.^(11:-1:0)));
	pbits(1+k) = v;
	numones(1+k) = sum(x);
end

f = fopen('golaytable.cc', 'w');

fprintf(f, '// Automatically generated from make_golaytable.m\n');
fprintf(f, '\n');
fprintf(f, 'int golay_parity[] = {\n');
for k = 1:4095
	fprintf(f, '%d, ', pbits(k));
	if mod(k, 16) == 0
		fprintf(f, '\n');
	end
end
fprintf(f, '%d', pbits(4096));
fprintf(f, '};\n');
fprintf(f, '\n');
fprintf(f, 'int golay_numones[] = {\n');
for k = 1:4095
	fprintf(f, '%d, ', numones(k));
	if mod(k, 16) == 0
		fprintf(f, '\n');
	end
end
fprintf(f, '%d', numones(4096));
fprintf(f, '};\n');
fclose(f);

