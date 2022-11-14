rm = rmgen(2,6);
f = fopen('rm_2_6.rm', 'w');
writermconf(rm, 'rm_2_6', f);
fclose(f);

