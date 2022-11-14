// megafunction wizard: %PARALLEL_ADD%VBB%
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
