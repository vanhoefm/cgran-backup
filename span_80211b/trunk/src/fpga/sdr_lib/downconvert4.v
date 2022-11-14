

module downconvert4(clk, strobe_in ,reset, datain, dataout, pul);
input clk, reset, strobe_in;
input [15:0] datain;
output reg [15:0] dataout;
output wire pul;


assign pul = ~|counter && strobe_in;

reg [2:0] counter;

/*always @(posedge clk)
begin
   if (strobe_in)
   begin
      dataout <= datain;
      pul <= 1;
   end
   else
     pul <= 0;
end
*/


always @(posedge clk)
begin
   if (reset)
      counter <= #1 0;
   else if (strobe_in)
      counter <= counter+1;
   /*else
      counter <= counter;*/
end


always @(posedge clk)
begin
   if ((counter==2'b000) && (strobe_in==1'b1))
   begin
      dataout <= datain;
   end
/*   else
   begin
      //dataout <= dataout;
      pul <= 0;
   end*/
end


endmodule
