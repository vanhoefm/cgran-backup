module shift_register(input clk, input reset, input rxstrobe, 
                 input [7:0] in_sample, output wire [7:0] out_sample, 
                 input [2:0] sel, output wire [7:0] data);
  
    reg signed [7:0] shift [5:0];
    integer i;
    assign out_sample = shift[5];
    assign data = (sel[2:1] == 2'd1) ? ((sel[0]) ? shift[1] : shift[0]) :
                  (sel[2:1] == 2'd2) ? ((sel[0]) ? shift[3] : shift[2]) :
                                       ((sel[0]) ? shift[5] : shift[4]) ;

    always @ (posedge clk)
        if(reset)
          begin
            for (i = 0; i< 6; i = i + 1)
                shift[i] <= 0;
          end
        else if (rxstrobe)
          begin
            for (i = 1; i< 6; i = i + 1)
                shift[i] <= shift[i-1];
            shift[0] <= in_sample;
          end

endmodule
