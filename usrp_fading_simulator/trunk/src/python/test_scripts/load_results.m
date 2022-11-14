function load_results(file_1, file_2, samples)
    a = read_float_binary(file_1, samples);
    b = read_float_binary(file_2, samples);
    plot(a);
    hold on;
    plot(b, 'r');