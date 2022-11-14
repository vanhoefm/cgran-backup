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

//this module is developed for when data rate is 32MSps, it only works correctly with this data rate

module FIR32(clk, reset, indata, data_conv, strobe_in);
input clk, reset, strobe_in;
input [15:0] indata;
output reg [15:0] data_conv;

integer n;

reg [15:0] mem [0:31];
reg [15:0] bmem[0:31];


reg [15:0] 	  mul16, mul15, mul14, mul13, mul11, mul9, mul6, mul5, mul3,
			  mul17, mul18, mul19, mul20, mul21, mul28, mul31;

wire [23:0] allsum1, allsum2, allsum3, allsum4;
wire [25:0] data; //for one clock pulse delay in pipeline

mult_add mult_add1 (
	.clock0(clk),
	.dataa_0(mul16),
	.dataa_1(mul15),
	.dataa_2(mul14),
	.dataa_3(mul13),
	.datab_0(6'd16),
	.datab_1(6'd15),
	.datab_2(6'd14),
	.datab_3(6'd13),
	.ena0(strobe_in),			//active high
	.result(allsum1));

mult_add mult_add2 (
	.clock0(clk),
	.dataa_0(mul11),
	.dataa_1(mul9),
	.dataa_2(mul6),
	.dataa_3(mul5),
	.datab_0(6'd11),
	.datab_1(6'd9),
	.datab_2(6'd6),
	.datab_3(6'd5),
	.ena0(strobe_in),
	.result(allsum2));

mult_add mult_add3 (
	.clock0(clk),
	.dataa_0(mul3),
	.dataa_1(mul17),
	.dataa_2(mul18),
	.dataa_3(mul19),
	.datab_0(6'd3),
	.datab_1(6'd17),
	.datab_2(6'd18),
	.datab_3(6'd19),
	.ena0(strobe_in),
	.result(allsum3));
	
mult_add mult_add4 (
	.clock0(clk),
	.dataa_0(mul20),
	.dataa_1(indata),
	.dataa_2(mul28),
	.dataa_3(mul31),
	.datab_0(6'd20),
	.datab_1(6'h30),
	.datab_2(6'd28),
	.datab_3(6'd31),
	.ena0(strobe_in),
	.result(allsum4));

padd par_add(
	.data0x(allsum1),
	.data1x(allsum2),
	.data2x(allsum3),
	.data3x(allsum4),
	.result(data));

always @(posedge clk)
begin
   if (strobe_in == 1'b1)
   begin
     data_conv <= data[25:10];
	 for (n=31; n > 0; n=n-1)
		bmem[n] = mem[n];
        bmem[0] <= indata;	//indata is valid only when strobe_in is asserted
   end
   else  //strobe_in
   begin
      for (n=31; n > 0; n=n-1)
		mem[n] = bmem[n-1];
		
      //remember that mem is shifted version of bmem
	  mul16 <= bmem[30]+bmem[12]-(bmem[2]+bmem[0]);
	  mul15 <= -(bmem[27]+bmem[1]);
	  mul14 <= bmem[8];
	  mul13 <= bmem[24]+bmem[15]-bmem[16];
	  mul11 <= -bmem[17];
	  mul9 <= bmem[29]-bmem[6];
	  mul6 <= -bmem[25];
	  mul5 <= -bmem[28];
	  mul3 <= bmem[20]+bmem[7]+bmem[16];	//bmem[16] should be multiplied by 2
	  mul17 <= bmem[13]+bmem[11]-(bmem[26]+bmem[18]+bmem[5]);
	  mul18 <= bmem[14]-bmem[3];
	  mul19 <= bmem[10]+bmem[9];
	  mul20 <= bmem[21]-bmem[4];	//bmem[21] should be multiplied by 21
	  mul28 <= bmem[23];
	  mul31 <= bmem[22];
   end
end


endmodule
