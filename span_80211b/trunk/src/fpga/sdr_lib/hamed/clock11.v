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


//counts from 31 to zero
//be careful that the strobe length should be one clock pulse, othewise you will have overrun error
module clock11(clk, reset, strobe_in, out, counter);
input clk;
input reset;
input strobe_in;
output out;
output reg [4:0] counter;

reg out_clk;

assign out = ~|counter && out_clk;

always @(posedge clk)
begin
   if (reset == 1'b1)
	 counter <= 5'b11111;
   else if ((counter == 5'b0) && (out_clk == 1))
		out_clk <= 0;
   else if (strobe_in)
   begin
	 counter <= counter - 5'd1;
	 out_clk <= 1;
   end
end


endmodule
