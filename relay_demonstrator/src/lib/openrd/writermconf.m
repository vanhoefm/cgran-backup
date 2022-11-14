function writermconf(rm, name, f)
%writermconf(rm, name, f)

for y = 1:rm.k
	fprintf(f, 'static char %s_G_%d[] = {', name, y-1);
	s = '';
	for x = 1:rm.n
		fprintf(f, '%s%d', s, rm.G(y,x));
		s = ', ';
	end
	fprintf(f, '};\n');
end

fprintf(f, 'static char* %s_G[] = {', name);
s = '';
for y = 1:rm.k
	fprintf(f, '%s%s_G_%d', s, name, y-1);
	s = ', ';
end
fprintf(f, '};\n');

fprintf(f, 'static int %s_mlvsize[] = {', name);
s = '';
for x = 1:length(rm.mlv)
	fprintf(f, '%s%d', s, size(rm.mlv{x}, 3));
	s = ', ';
end
fprintf(f, '};\n');


for x1 = 1:length(rm.mlv)
	v = rm.mlv{x1};
	for x2 = 1:size(v, 3)
		for x3 = 1:size(v, 1)
			fprintf(f, 'static unsigned short %s_mlv_%d_%d_%d[] = {', name, x1-1, x2-1, x3-1);
			s = '';
			for x4 = 1:size(v, 2)
				fprintf(f, '%s%d', s, v(x3, x4, x2));
				s = ', ';
			end
			fprintf(f, '};\n');
		end

		fprintf(f, 'static unsigned short* %s_mlv_%d_%d[] = {', name, x1-1, x2-1);
		s = '';
		for x3 = 1:size(v, 1)
			fprintf(f, '%s%s_mlv_%d_%d_%d', s, name, x1-1, x2-1, x3-1);
			s = ', ';
		end
		fprintf(f, '};\n');
	end

	fprintf(f, 'static unsigned short** %s_mlv_%d[] = {', name, x1-1);
	s = '';
	for x2 = 1:size(v, 3)
		fprintf(f, '%s%s_mlv_%d_%d', s, name, x1-1, x2-1);
		s = ', ';
	end
	fprintf(f, '};\n');
end

fprintf(f, 'static unsigned short*** %s_mlv[] = {', name);
s = '';
for x = 1:length(rm.mlv)
	fprintf(f, '%s%s_mlv_%d', s, name, x-1);
	s = ', ';
end
fprintf(f, '};\n');

fprintf(f, 'static const rm_t %s = {%d, %d, %d, %d, %s_G, %s_mlvsize, %s_mlv};\n', ...
	name, rm.r, rm.m, rm.k, rm.n, name, name, name);

return

fprintf(f, '  {\n');
s1 = '';
for y = 1:rm.k
	fprintf(f, '%s    {', s1);
	s2 = '';
	for x = 1:rm.n
		fprintf(f, '%s%d', s2, rm.G(y,x));
		s2 = ', ';
	end
	fprintf(f, '}');
	s1 = sprintf(',\n');
end
fprintf(f, '\n');
fprintf(f, '  },\n');
fprintf(f, '  {');
s1 = '';
for x = 1:length(rm.mlv)
	fprintf(f, '%s%d', s1, size(rm.mlv{x}, 3));
	s1 = ', ';
end
fprintf(f, '},\n');
fprintf(f, '  {\n');
s1 = '';
for x1 = 1:length(rm.mlv)
	v = rm.mlv{x1};
	fprintf(f, '%s    {\n', s1);
	s1 = sprintf(',\n');
	s2 = '';
	for x2 = 1:size(v, 3)
		fprintf(f, '%s      {', s2);
		s2 = sprintf(',\n');
		s3 = '';
		for x3 = 1:size(v, 1)
			fprintf(f, '%s{', s3);
			s3 = ', ';
			s4 = '';
			for x4 = 1:size(v, 2)
				fprintf(f, '%s%d', s4, v(x3,x4,x2));
				s4 = ', ';
			end
			fprintf(f, '}');
		end
		fprintf(f, '}');
	end
	fprintf(f, '\n    }');
end
fprintf(f, '\n  }\n');
fprintf(f, '}\n');

