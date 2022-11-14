// megafunction wizard: %LPM_ADD_SUB%CBX%
// GENERATION: STANDARD
// VERSION: WM1.0
// MODULE: lpm_add_sub 

// ============================================================
// File Name: addsub16.v
// Megafunction Name(s):
// 			lpm_add_sub
// ============================================================
// ************************************************************
// THIS IS A WIZARD-GENERATED FILE. DO NOT EDIT THIS FILE!
// ************************************************************


//Copyright (C) 1991-2003 Altera Corporation
//Any  megafunction  design,  and related netlist (encrypted  or  decrypted),
//support information,  device programming or simulation file,  and any other
//associated  documentation or information  provided by  Altera  or a partner
//under  Altera's   Megafunction   Partnership   Program  may  be  used  only
//to program  PLD  devices (but not masked  PLD  devices) from  Altera.   Any
//other  use  of such  megafunction  design,  netlist,  support  information,
//device programming or simulation file,  or any other  related documentation
//or information  is prohibited  for  any  other purpose,  including, but not
//limited to  modification,  reverse engineering,  de-compiling, or use  with
//any other  silicon devices,  unless such use is  explicitly  licensed under
//a separate agreement with  Altera  or a megafunction partner.  Title to the
//intellectual property,  including patents,  copyrights,  trademarks,  trade
//secrets,  or maskworks,  embodied in any such megafunction design, netlist,
//support  information,  device programming or simulation file,  or any other
//related documentation or information provided by  Altera  or a megafunction
//partner, remains with Altera, the megafunction partner, or their respective
//licensors. No other licenses, including any licenses needed under any third
//party's intellectual property, are provided herein.


//lpm_add_sub DEVICE_FAMILY=Cyclone LPM_PIPELINE=1 LPM_WIDTH=16 aclr add_sub clken clock dataa datab result
//VERSION_BEGIN 3.0 cbx_lpm_add_sub 2003:04:10:18:28:42:SJ cbx_mgl 2003:06:11:11:00:44:SJ cbx_stratix 2003:05:16:10:26:50:SJ  VERSION_END

//synthesis_resources = lut 17 
module  addsub16_add_sub_gp9
	( 
	aclr,
	add_sub,
	clken,
	clock,
	dataa,
	datab,
	result) /* synthesis synthesis_clearbox=1 */;
	input   aclr;
	input   add_sub;
	input   clken;
	input   clock;
	input   [15:0]  dataa;
	input   [15:0]  datab;
	output   [15:0]  result;

	wire  [0:0]   wire_add_sub_cella_0cout;
	wire  [0:0]   wire_add_sub_cella_1cout;
	wire  [0:0]   wire_add_sub_cella_2cout;
	wire  [0:0]   wire_add_sub_cella_3cout;
	wire  [0:0]   wire_add_sub_cella_4cout;
	wire  [0:0]   wire_add_sub_cella_5cout;
	wire  [0:0]   wire_add_sub_cella_6cout;
	wire  [0:0]   wire_add_sub_cella_7cout;
	wire  [0:0]   wire_add_sub_cella_8cout;
	wire  [0:0]   wire_add_sub_cella_9cout;
	wire  [0:0]   wire_add_sub_cella_10cout;
	wire  [0:0]   wire_add_sub_cella_11cout;
	wire  [0:0]   wire_add_sub_cella_12cout;
	wire  [0:0]   wire_add_sub_cella_13cout;
	wire  [0:0]   wire_add_sub_cella_14cout;
	wire  [15:0]   wire_add_sub_cella_dataa;
	wire  [15:0]   wire_add_sub_cella_datab;
	wire  [15:0]   wire_add_sub_cella_regout;
	wire  wire_strx_lcell1_cout;

	stratix_lcell   add_sub_cella_0
	( 
	.aclr(aclr),
	.cin(wire_strx_lcell1_cout),
	.clk(clock),
	.cout(wire_add_sub_cella_0cout[0:0]),
	.dataa(wire_add_sub_cella_dataa[0:0]),
	.datab(wire_add_sub_cella_datab[0:0]),
	.ena(clken),
	.inverta((~ add_sub)),
	.regout(wire_add_sub_cella_regout[0:0]));
	defparam
		add_sub_cella_0.cin_used = "true",
		add_sub_cella_0.lut_mask = "96e8",
		add_sub_cella_0.operation_mode = "arithmetic",
		add_sub_cella_0.sum_lutc_input = "cin",
		add_sub_cella_0.lpm_type = "stratix_lcell";
	stratix_lcell   add_sub_cella_1
	( 
	.aclr(aclr),
	.cin(wire_add_sub_cella_0cout[0:0]),
	.clk(clock),
	.cout(wire_add_sub_cella_1cout[0:0]),
	.dataa(wire_add_sub_cella_dataa[1:1]),
	.datab(wire_add_sub_cella_datab[1:1]),
	.ena(clken),
	.inverta((~ add_sub)),
	.regout(wire_add_sub_cella_regout[1:1]));
	defparam
		add_sub_cella_1.cin_used = "true",
		add_sub_cella_1.lut_mask = "96e8",
		add_sub_cella_1.operation_mode = "arithmetic",
		add_sub_cella_1.sum_lutc_input = "cin",
		add_sub_cella_1.lpm_type = "stratix_lcell";
	stratix_lcell   add_sub_cella_2
	( 
	.aclr(aclr),
	.cin(wire_add_sub_cella_1cout[0:0]),
	.clk(clock),
	.cout(wire_add_sub_cella_2cout[0:0]),
	.dataa(wire_add_sub_cella_dataa[2:2]),
	.datab(wire_add_sub_cella_datab[2:2]),
	.ena(clken),
	.inverta((~ add_sub)),
	.regout(wire_add_sub_cella_regout[2:2]));
	defparam
		add_sub_cella_2.cin_used = "true",
		add_sub_cella_2.lut_mask = "96e8",
		add_sub_cella_2.operation_mode = "arithmetic",
		add_sub_cella_2.sum_lutc_input = "cin",
		add_sub_cella_2.lpm_type = "stratix_lcell";
	stratix_lcell   add_sub_cella_3
	( 
	.aclr(aclr),
	.cin(wire_add_sub_cella_2cout[0:0]),
	.clk(clock),
	.cout(wire_add_sub_cella_3cout[0:0]),
	.dataa(wire_add_sub_cella_dataa[3:3]),
	.datab(wire_add_sub_cella_datab[3:3]),
	.ena(clken),
	.inverta((~ add_sub)),
	.regout(wire_add_sub_cella_regout[3:3]));
	defparam
		add_sub_cella_3.cin_used = "true",
		add_sub_cella_3.lut_mask = "96e8",
		add_sub_cella_3.operation_mode = "arithmetic",
		add_sub_cella_3.sum_lutc_input = "cin",
		add_sub_cella_3.lpm_type = "stratix_lcell";
	stratix_lcell   add_sub_cella_4
	( 
	.aclr(aclr),
	.cin(wire_add_sub_cella_3cout[0:0]),
	.clk(clock),
	.cout(wire_add_sub_cella_4cout[0:0]),
	.dataa(wire_add_sub_cella_dataa[4:4]),
	.datab(wire_add_sub_cella_datab[4:4]),
	.ena(clken),
	.inverta((~ add_sub)),
	.regout(wire_add_sub_cella_regout[4:4]));
	defparam
		add_sub_cella_4.cin_used = "true",
		add_sub_cella_4.lut_mask = "96e8",
		add_sub_cella_4.operation_mode = "arithmetic",
		add_sub_cella_4.sum_lutc_input = "cin",
		add_sub_cella_4.lpm_type = "stratix_lcell";
	stratix_lcell   add_sub_cella_5
	( 
	.aclr(aclr),
	.cin(wire_add_sub_cella_4cout[0:0]),
	.clk(clock),
	.cout(wire_add_sub_cella_5cout[0:0]),
	.dataa(wire_add_sub_cella_dataa[5:5]),
	.datab(wire_add_sub_cella_datab[5:5]),
	.ena(clken),
	.inverta((~ add_sub)),
	.regout(wire_add_sub_cella_regout[5:5]));
	defparam
		add_sub_cella_5.cin_used = "true",
		add_sub_cella_5.lut_mask = "96e8",
		add_sub_cella_5.operation_mode = "arithmetic",
		add_sub_cella_5.sum_lutc_input = "cin",
		add_sub_cella_5.lpm_type = "stratix_lcell";
	stratix_lcell   add_sub_cella_6
	( 
	.aclr(aclr),
	.cin(wire_add_sub_cella_5cout[0:0]),
	.clk(clock),
	.cout(wire_add_sub_cella_6cout[0:0]),
	.dataa(wire_add_sub_cella_dataa[6:6]),
	.datab(wire_add_sub_cella_datab[6:6]),
	.ena(clken),
	.inverta((~ add_sub)),
	.regout(wire_add_sub_cella_regout[6:6]));
	defparam
		add_sub_cella_6.cin_used = "true",
		add_sub_cella_6.lut_mask = "96e8",
		add_sub_cella_6.operation_mode = "arithmetic",
		add_sub_cella_6.sum_lutc_input = "cin",
		add_sub_cella_6.lpm_type = "stratix_lcell";
	stratix_lcell   add_sub_cella_7
	( 
	.aclr(aclr),
	.cin(wire_add_sub_cella_6cout[0:0]),
	.clk(clock),
	.cout(wire_add_sub_cella_7cout[0:0]),
	.dataa(wire_add_sub_cella_dataa[7:7]),
	.datab(wire_add_sub_cella_datab[7:7]),
	.ena(clken),
	.inverta((~ add_sub)),
	.regout(wire_add_sub_cella_regout[7:7]));
	defparam
		add_sub_cella_7.cin_used = "true",
		add_sub_cella_7.lut_mask = "96e8",
		add_sub_cella_7.operation_mode = "arithmetic",
		add_sub_cella_7.sum_lutc_input = "cin",
		add_sub_cella_7.lpm_type = "stratix_lcell";
	stratix_lcell   add_sub_cella_8
	( 
	.aclr(aclr),
	.cin(wire_add_sub_cella_7cout[0:0]),
	.clk(clock),
	.cout(wire_add_sub_cella_8cout[0:0]),
	.dataa(wire_add_sub_cella_dataa[8:8]),
	.datab(wire_add_sub_cella_datab[8:8]),
	.ena(clken),
	.inverta((~ add_sub)),
	.regout(wire_add_sub_cella_regout[8:8]));
	defparam
		add_sub_cella_8.cin_used = "true",
		add_sub_cella_8.lut_mask = "96e8",
		add_sub_cella_8.operation_mode = "arithmetic",
		add_sub_cella_8.sum_lutc_input = "cin",
		add_sub_cella_8.lpm_type = "stratix_lcell";
	stratix_lcell   add_sub_cella_9
	( 
	.aclr(aclr),
	.cin(wire_add_sub_cella_8cout[0:0]),
	.clk(clock),
	.cout(wire_add_sub_cella_9cout[0:0]),
	.dataa(wire_add_sub_cella_dataa[9:9]),
	.datab(wire_add_sub_cella_datab[9:9]),
	.ena(clken),
	.inverta((~ add_sub)),
	.regout(wire_add_sub_cella_regout[9:9]));
	defparam
		add_sub_cella_9.cin_used = "true",
		add_sub_cella_9.lut_mask = "96e8",
		add_sub_cella_9.operation_mode = "arithmetic",
		add_sub_cella_9.sum_lutc_input = "cin",
		add_sub_cella_9.lpm_type = "stratix_lcell";
	stratix_lcell   add_sub_cella_10
	( 
	.aclr(aclr),
	.cin(wire_add_sub_cella_9cout[0:0]),
	.clk(clock),
	.cout(wire_add_sub_cella_10cout[0:0]),
	.dataa(wire_add_sub_cella_dataa[10:10]),
	.datab(wire_add_sub_cella_datab[10:10]),
	.ena(clken),
	.inverta((~ add_sub)),
	.regout(wire_add_sub_cella_regout[10:10]));
	defparam
		add_sub_cella_10.cin_used = "true",
		add_sub_cella_10.lut_mask = "96e8",
		add_sub_cella_10.operation_mode = "arithmetic",
		add_sub_cella_10.sum_lutc_input = "cin",
		add_sub_cella_10.lpm_type = "stratix_lcell";
	stratix_lcell   add_sub_cella_11
	( 
	.aclr(aclr),
	.cin(wire_add_sub_cella_10cout[0:0]),
	.clk(clock),
	.cout(wire_add_sub_cella_11cout[0:0]),
	.dataa(wire_add_sub_cella_dataa[11:11]),
	.datab(wire_add_sub_cella_datab[11:11]),
	.ena(clken),
	.inverta((~ add_sub)),
	.regout(wire_add_sub_cella_regout[11:11]));
	defparam
		add_sub_cella_11.cin_used = "true",
		add_sub_cella_11.lut_mask = "96e8",
		add_sub_cella_11.operation_mode = "arithmetic",
		add_sub_cella_11.sum_lutc_input = "cin",
		add_sub_cella_11.lpm_type = "stratix_lcell";
	stratix_lcell   add_sub_cella_12
	( 
	.aclr(aclr),
	.cin(wire_add_sub_cella_11cout[0:0]),
	.clk(clock),
	.cout(wire_add_sub_cella_12cout[0:0]),
	.dataa(wire_add_sub_cella_dataa[12:12]),
	.datab(wire_add_sub_cella_datab[12:12]),
	.ena(clken),
	.inverta((~ add_sub)),
	.regout(wire_add_sub_cella_regout[12:12]));
	defparam
		add_sub_cella_12.cin_used = "true",
		add_sub_cella_12.lut_mask = "96e8",
		add_sub_cella_12.operation_mode = "arithmetic",
		add_sub_cella_12.sum_lutc_input = "cin",
		add_sub_cella_12.lpm_type = "stratix_lcell";
	stratix_lcell   add_sub_cella_13
	( 
	.aclr(aclr),
	.cin(wire_add_sub_cella_12cout[0:0]),
	.clk(clock),
	.cout(wire_add_sub_cella_13cout[0:0]),
	.dataa(wire_add_sub_cella_dataa[13:13]),
	.datab(wire_add_sub_cella_datab[13:13]),
	.ena(clken),
	.inverta((~ add_sub)),
	.regout(wire_add_sub_cella_regout[13:13]));
	defparam
		add_sub_cella_13.cin_used = "true",
		add_sub_cella_13.lut_mask = "96e8",
		add_sub_cella_13.operation_mode = "arithmetic",
		add_sub_cella_13.sum_lutc_input = "cin",
		add_sub_cella_13.lpm_type = "stratix_lcell";
	stratix_lcell   add_sub_cella_14
	( 
	.aclr(aclr),
	.cin(wire_add_sub_cella_13cout[0:0]),
	.clk(clock),
	.cout(wire_add_sub_cella_14cout[0:0]),
	.dataa(wire_add_sub_cella_dataa[14:14]),
	.datab(wire_add_sub_cella_datab[14:14]),
	.ena(clken),
	.inverta((~ add_sub)),
	.regout(wire_add_sub_cella_regout[14:14]));
	defparam
		add_sub_cella_14.cin_used = "true",
		add_sub_cella_14.lut_mask = "96e8",
		add_sub_cella_14.operation_mode = "arithmetic",
		add_sub_cella_14.sum_lutc_input = "cin",
		add_sub_cella_14.lpm_type = "stratix_lcell";
	stratix_lcell   add_sub_cella_15
	( 
	.aclr(aclr),
	.cin(wire_add_sub_cella_14cout[0:0]),
	.clk(clock),
	.dataa(wire_add_sub_cella_dataa[15:15]),
	.datab(wire_add_sub_cella_datab[15:15]),
	.ena(clken),
	.inverta((~ add_sub)),
	.regout(wire_add_sub_cella_regout[15:15]));
	defparam
		add_sub_cella_15.cin_used = "true",
		add_sub_cella_15.lut_mask = "9696",
		add_sub_cella_15.operation_mode = "normal",
		add_sub_cella_15.sum_lutc_input = "cin",
		add_sub_cella_15.lpm_type = "stratix_lcell";
	assign
		wire_add_sub_cella_dataa = datab,
		wire_add_sub_cella_datab = dataa;
	stratix_lcell   strx_lcell1
	( 
	.cout(wire_strx_lcell1_cout),
	.dataa(1'b0),
	.datab((~ add_sub)),
	.inverta((~ add_sub)));
	defparam
		strx_lcell1.cin_used = "false",
		strx_lcell1.lut_mask = "00cc",
		strx_lcell1.operation_mode = "arithmetic",
		strx_lcell1.lpm_type = "stratix_lcell";
	assign
		result = wire_add_sub_cella_regout;
endmodule //addsub16_add_sub_gp9
//VALID FILE


module addsub16 (
	add_sub,
	dataa,
	datab,
	clock,
	aclr,
	clken,
	result)/* synthesis synthesis_clearbox = 1 */;

	input	  add_sub;
	input	[15:0]  dataa;
	input	[15:0]  datab;
	input	  clock;
	input	  aclr;
	input	  clken;
	output	[15:0]  result;

	wire [15:0] sub_wire0;
	wire [15:0] result = sub_wire0[15:0];

	addsub16_add_sub_gp9	addsub16_add_sub_gp9_component (
				.dataa (dataa),
				.add_sub (add_sub),
				.datab (datab),
				.clken (clken),
				.aclr (aclr),
				.clock (clock),
				.result (sub_wire0));

endmodule

// ============================================================
// CNX file retrieval info
// ============================================================
// Retrieval info: PRIVATE: nBit NUMERIC "16"
// Retrieval info: PRIVATE: Function NUMERIC "2"
// Retrieval info: PRIVATE: WhichConstant NUMERIC "0"
// Retrieval info: PRIVATE: ConstantA NUMERIC "0"
// Retrieval info: PRIVATE: ConstantB NUMERIC "0"
// Retrieval info: PRIVATE: ValidCtA NUMERIC "0"
// Retrieval info: PRIVATE: ValidCtB NUMERIC "0"
// Retrieval info: PRIVATE: CarryIn NUMERIC "0"
// Retrieval info: PRIVATE: CarryOut NUMERIC "0"
// Retrieval info: PRIVATE: Overflow NUMERIC "0"
// Retrieval info: PRIVATE: Latency NUMERIC "1"
// Retrieval info: PRIVATE: aclr NUMERIC "1"
// Retrieval info: PRIVATE: clken NUMERIC "1"
// Retrieval info: PRIVATE: LPM_PIPELINE NUMERIC "1"
// Retrieval info: PRIVATE: INTENDED_DEVICE_FAMILY STRING "Cyclone"
// Retrieval info: CONSTANT: LPM_WIDTH NUMERIC "16"
// Retrieval info: CONSTANT: LPM_DIRECTION STRING "UNUSED"
// Retrieval info: CONSTANT: LPM_TYPE STRING "LPM_ADD_SUB"
// Retrieval info: CONSTANT: LPM_HINT STRING "ONE_INPUT_IS_CONSTANT=NO"
// Retrieval info: CONSTANT: LPM_PIPELINE NUMERIC "1"
// Retrieval info: CONSTANT: INTENDED_DEVICE_FAMILY STRING "Cyclone"
// Retrieval info: USED_PORT: add_sub 0 0 0 0 INPUT NODEFVAL add_sub
// Retrieval info: USED_PORT: result 0 0 16 0 OUTPUT NODEFVAL result[15..0]
// Retrieval info: USED_PORT: dataa 0 0 16 0 INPUT NODEFVAL dataa[15..0]
// Retrieval info: USED_PORT: datab 0 0 16 0 INPUT NODEFVAL datab[15..0]
// Retrieval info: USED_PORT: clock 0 0 0 0 INPUT NODEFVAL clock
// Retrieval info: USED_PORT: aclr 0 0 0 0 INPUT NODEFVAL aclr
// Retrieval info: USED_PORT: clken 0 0 0 0 INPUT NODEFVAL clken
// Retrieval info: CONNECT: @add_sub 0 0 0 0 add_sub 0 0 0 0
// Retrieval info: CONNECT: result 0 0 16 0 @result 0 0 16 0
// Retrieval info: CONNECT: @dataa 0 0 16 0 dataa 0 0 16 0
// Retrieval info: CONNECT: @datab 0 0 16 0 datab 0 0 16 0
// Retrieval info: CONNECT: @clock 0 0 0 0 clock 0 0 0 0
// Retrieval info: CONNECT: @aclr 0 0 0 0 aclr 0 0 0 0
// Retrieval info: CONNECT: @clken 0 0 0 0 clken 0 0 0 0
// Retrieval info: LIBRARY: lpm lpm.lpm_components.all
