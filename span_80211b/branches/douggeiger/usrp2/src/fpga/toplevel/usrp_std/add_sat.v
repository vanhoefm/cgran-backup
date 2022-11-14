


module add_sat(data1, data2, sum);
input [23:0] data1, data2;
output [23:0] sum;

wire [23:0] tsum;

wire cout, OF;

alt_add add(
	.dataa(data1),
	.datab(data2),
	.cout(cout),
	.overflow(OF),
	.result(tsum));

assign sum = (OF==0)?tsum:(cout)?24'h800000:24'h7FFFFF;

endmodule
