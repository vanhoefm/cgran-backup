module match_filter_test();

  reg rxclk;
  reg reset;
  reg [15:0] ch_0;
  reg [15:0] ch_1;
  wire rxstrobe;
  reg [31:0] cdata;
  reg [2:0] cstate;
  reg cwrite;
  wire valid;
  wire match;
  integer file_co, file_dat; 
  reg [7:0] data0;
  reg [7:0] data1;
  reg [7:0] data2;
  reg [7:0] data3;

  wire [15:0] final_result;
  reg [31:0] count;

  match_filter mf
   (.clk(rxclk), .reset(reset), .r_input(ch_0), .i_input(ch_1), 
    .rxstrobe(rxstrobe), .cdata(cdata),  .cstate(cstate), .cwrite(cwrite), 
    .valid(valid), .match(match), .debugbus(final_result));
  
  strobe_gen sgen
   (.reset(reset), .enable(1'b1), .clock(rxclk), .strobe_in(1'b1),
    .strobe(rxstrobe), .rate(8'd16));

  always
      #5 rxclk = ~rxclk ;

  initial
    begin
      reset   = 1;
      rxclk   = 0;
      ch_0    = 0;
      ch_1    = 0;
      cdata   = 0;
      cstate = 0;
      cwrite  = 0;
      count  = 0;
      
      #40 reset = 1'b0;

      file_co  = $fopen("z:/fpga/inband_lib/simulation/tx_cs.dat", "rb");

      if(!file_co) begin
        $display("Error opening coefficients\n");
      end

      repeat (3)
      begin
        $fread(data0, file_co);
        $fread(data1, file_co);
        $fread(data2, file_co);
        $fread(data3, file_co);
      end
      cdata [7:0] = data0;
        $fread(data0, file_co);
        $fread(data1, file_co);
        $fread(data2, file_co);
        $fread(data3, file_co);
      cdata [31:24] = data1;
      cdata [23:16] = data0;
      cdata [15:8]  = 8'd0;
      
      begin
      @(posedge rxclk)
        cwrite = 1;
        cstate = 1;
      end

      repeat (6)
      begin
      @(posedge rxclk)
        cwrite = 1;
        cstate = cstate + 1;
        $fread(data0, file_co);
        $fread(data1, file_co);
        $fread(data2, file_co);
        $fread(data3, file_co);
        cdata = {data3, data2, data1, data0}; 
      end

      $fclose(file_co);
     
      @(posedge rxclk)
        cwrite = 0;

      file_dat = $fopen("z:/fpga/inband_lib/simulation/rx_data_match.dat", "rb");
      if(!file_dat) begin
        $display("Error opening data\n");
      end
      
      while ($feof(file_dat) == 0)
        begin
          @(posedge rxstrobe)
            $fread(data0, file_dat);
            $fread(data1, file_dat);
            $fread(data2, file_dat);
            $fread(data3, file_dat);
            //$display("Reading in data %d, %d, %d, %d\n", data0, data1, data2, data3);
            ch_0 = {data1, data0};
            ch_1 = {data3, data2};
            count = count + 1;
            if (count > 96)
            	$display("count: %d, final: %d\n", count-32'd96, final_result);
        end 
      $fclose(file_dat);  
    end
endmodule

