// megafunction wizard: %PARALLEL_ADD%
// GENERATION: STANDARD
// VERSION: WM1.0
// MODULE: parallel_add 

// ============================================================
// File Name: padd.v
// Megafunction Name(s):
// 			parallel_add
//
// Simulation Library Files(s):
// 			altera_mf
// ============================================================
// ************************************************************
// THIS IS A WIZARD-GENERATED FILE. DO NOT EDIT THIS FILE!
//
// 7.2 Build 175 11/20/2007 SP 1 SJ Web Edition
// ************************************************************


//Copyright (C) 1991-2007 Altera Corporation
//Your use of Altera Corporation's design tools, logic functions 
//and other software and tools, and its AMPP partner logic 
//functions, and any output files from any of the foregoing 
//(including device programming or simulation files), and any 
//associated documentation or information are expressly subject 
//to the terms and conditions of the Altera Program License 
//Subscription Agreement, Altera MegaCore Function License 
//Agreement, or other applicable license agreement, including, 
//without limitation, that your use is for the sole purpose of 
//programming logic devices manufactured by Altera and sold by 
//Altera or its authorized distributors.  Please refer to the 
//applicable agreement for further details.


// synopsys translate_off
`timescale 1 ps / 1 ps
// synopsys translate_on
module padd (
	data0x,
	data1x,
	data2x,
	data3x,
	result);

	input	[23:0]  data0x;
	input	[23:0]  data1x;
	input	[23:0]  data2x;
	input	[23:0]  data3x;
	output	[25:0]  result;

	wire [25:0] sub_wire0;
	wire [23:0] sub_wire5 = data3x[23:0];
	wire [23:0] sub_wire4 = data1x[23:0];
	wire [23:0] sub_wire3 = data0x[23:0];
	wire [25:0] result = sub_wire0[25:0];
	wire [23:0] sub_wire1 = data2x[23:0];
	wire [95:0] sub_wire2 = {sub_wire5, sub_wire1, sub_wire4, sub_wire3};

	parallel_add	parallel_add_component (
				.data (sub_wire2),
				.result (sub_wire0)
				// synopsys translate_off
				,
				.aclr (),
				.clken (),
				.clock ()
				// synopsys translate_on
				);
	defparam
		parallel_add_component.msw_subtract = "NO",
		parallel_add_component.pipeline = 0,
		parallel_add_component.representation = "SIGNED",
		parallel_add_component.result_alignment = "LSB",
		parallel_add_component.shift = 0,
		parallel_add_component.size = 4,
		parallel_add_component.width = 24,
		parallel_add_component.widthr = 26;


endmodule

// ============================================================
// CNX file retrieval info
// ============================================================
// Retrieval info: PRIVATE: INTENDED_DEVICE_FAMILY STRING "Cyclone"
// Retrieval info: PRIVATE: SYNTH_WRAPPER_GEN_POSTFIX STRING "0"
// Retrieval info: LIBRARY: altera_mf altera_mf.altera_mf_components.all
// Retrieval info: CONSTANT: MSW_SUBTRACT STRING "NO"
// Retrieval info: CONSTANT: PIPELINE NUMERIC "0"
// Retrieval info: CONSTANT: REPRESENTATION STRING "SIGNED"
// Retrieval info: CONSTANT: RESULT_ALIGNMENT STRING "LSB"
// Retrieval info: CONSTANT: SHIFT NUMERIC "0"
// Retrieval info: CONSTANT: SIZE NUMERIC "4"
// Retrieval info: CONSTANT: WIDTH NUMERIC "24"
// Retrieval info: CONSTANT: WIDTHR NUMERIC "26"
// Retrieval info: USED_PORT: data0x 0 0 24 0 INPUT NODEFVAL "data0x[23..0]"
// Retrieval info: USED_PORT: data1x 0 0 24 0 INPUT NODEFVAL "data1x[23..0]"
// Retrieval info: USED_PORT: data2x 0 0 24 0 INPUT NODEFVAL "data2x[23..0]"
// Retrieval info: USED_PORT: data3x 0 0 24 0 INPUT NODEFVAL "data3x[23..0]"
// Retrieval info: USED_PORT: result 0 0 26 0 OUTPUT NODEFVAL "result[25..0]"
// Retrieval info: CONNECT: @data 0 0 24 72 data3x 0 0 24 0
// Retrieval info: CONNECT: @data 0 0 24 48 data2x 0 0 24 0
// Retrieval info: CONNECT: @data 0 0 24 24 data1x 0 0 24 0
// Retrieval info: CONNECT: @data 0 0 24 0 data0x 0 0 24 0
// Retrieval info: CONNECT: result 0 0 26 0 @result 0 0 26 0
// Retrieval info: GEN_FILE: TYPE_NORMAL padd.v TRUE FALSE
// Retrieval info: GEN_FILE: TYPE_NORMAL padd.inc FALSE FALSE
// Retrieval info: GEN_FILE: TYPE_NORMAL padd.cmp FALSE FALSE
// Retrieval info: GEN_FILE: TYPE_NORMAL padd.bsf FALSE FALSE
// Retrieval info: GEN_FILE: TYPE_NORMAL padd_inst.v FALSE FALSE
// Retrieval info: GEN_FILE: TYPE_NORMAL padd_bb.v TRUE FALSE
// Retrieval info: LIB_FILE: altera_mf