// -*- verilog -*-
//
//  Sensing and Processing Across Networks (SPAN) Lab.
//  University of Utah, UT-84112
//  802.11b receiver v1.0
//  Author: Mohammad H. Firooz   mhfirooz@yahoo.com
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 51 Franklin Street, Boston, MA  02110-1301  USA
//


module despreading(clk, reset, dataini, datainq, strobe_in, dataouti, dataoutq, strobe_out);
input clk, reset;
input [15:0] dataini, datainq;
input strobe_in;
output [15:0] dataouti, dataoutq;
output strobe_out;

wire pul;
wire [15:0] data_convi, data_convq;
wire [32:0] absdata;
wire [15:0] absi, absq, sumabs;


/*singleABS ABSi(
	.data(data_convi),
	.result(absi));
	
singleABS ABSq(
	.data(data_convq),
	.result(absq));

wire cout;

alt_add adder(
	.dataa(absi),
	.datab(absq),
	.cout(cout),
	.result(sumabs));

assign absdata = {cout, sumabs};
*/

ComplexAbs complexabs(.datai(data_convi), .dataq(data_convq), .absval(absdata));

wire strobe_out1;
wire [4:0] counter;
clock11 myclk(.clk(clk), .reset(reset), .strobe_in(strobe_in), .out(strobe_out), .counter(counter));
peakfinder peak(.clk(clk), .reset(reset), .absdata(regabsdata), .datai(regdata_convi), .dataq(regdata_convq), 
		    .datastrobe(strobe_in), .maxstrobe(strobe_out), .maxdatai(dataouti), .maxdataq(dataoutq), 
		    .counter(counter));

FIR32 FIRi(.clk(clk), .reset(reset), .indata(dataini), .data_conv(data_convi), .strobe_in(strobe_in));
FIR32 FIRq(.clk(clk), .reset(reset), .indata(datainq), .data_conv(data_convq), .strobe_in(strobe_in));	

reg [15:0] regdata_convi, regdata_convq;
reg [32:0] regabsdata;

always @(posedge clk)
begin
  if (strobe_in == 1'b1)
  begin
     regdata_convi <= data_convi;
     regdata_convq <= data_convq;
     regabsdata <= absdata;
  end
end

endmodule
