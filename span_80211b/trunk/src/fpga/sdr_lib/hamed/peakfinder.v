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


module peakfinder(clk, reset, absdata, datai, dataq, datastrobe, maxstrobe, maxdatai, maxdataq, counter,strobe_out);
input clk;
input reset;
input datastrobe;					//when asserted means input data is valid
input [32:0] absdata;		//input data
input [15:0] datai, dataq;
input maxstrobe;				//when asserted, 32 sample period is over
input [4:0] counter;
output [15:0] maxdatai, maxdataq;		
output reg strobe_out;

reg [32:0] max_data;
reg [15:0] max_datai, max_dataq;
reg reset_max;

assign maxdatai = max_datai;
assign maxdataq = max_dataq;

always @(posedge clk)
begin
	if (reset == 1'b1 || maxstrobe == 1'b1)
	begin
	   reset_max <= 1;
	end
	else if (reset_max == 1'b1)
	begin
		max_data <= absdata;
	    max_datai <= datai;
		max_dataq <= dataq;
		reset_max <= 0;
	end
	else if ((datastrobe == 1'b0) && (absdata > max_data))		
	begin
			max_data <= absdata;
			max_datai <= datai;
			max_dataq <= dataq;
	end //datastrobe			  
end
     
endmodule
