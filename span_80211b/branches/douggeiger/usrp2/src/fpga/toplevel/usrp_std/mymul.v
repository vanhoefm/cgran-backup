// megafunction wizard: %LPM_MULT%
// GENERATION: STANDARD
// VERSION: WM1.0
// MODULE: lpm_mult 

// ============================================================
// File Name: mymul.v
// Megafunction Name(s):
// 			lpm_mult
//
// Simulation Library Files(s):
// 			lpm
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
module mymul (
	dataa,
	result);

	input	[15:0]  dataa;
	output	[20:0]  result;

	wire [20:0] sub_wire0;
	wire [4:0] sub_wire1 = 5'h0a;
	wire [20:0] result = sub_wire0[20:0];

	lpm_mult	lpm_mult_component (
				.dataa (dataa),
				.datab (sub_wire1),
				.result (sub_wire0),
				.aclr (1'b0),
				.clken (1'b1),
				.clock (1'b0),
				.sum (1'b0));
	defparam
		lpm_mult_component.lpm_hint = "INPUT_B_IS_CONSTANT=YES,MAXIMIZE_SPEED=5",
		lpm_mult_component.lpm_representation = "SIGNED",
		lpm_mult_component.lpm_type = "LPM_MULT",
		lpm_mult_component.lpm_widtha = 16,
		lpm_mult_component.lpm_widthb = 5,
		lpm_mult_component.lpm_widthp = 21;


endmodule

// ============================================================
// CNX file retrieval info
// ============================================================
// Retrieval info: PRIVATE: AutoSizeResult NUMERIC "1"
// Retrieval info: PRIVATE: B_isConstant NUMERIC "1"
// Retrieval info: PRIVATE: ConstantB NUMERIC "10"
// Retrieval info: PRIVATE: INTENDED_DEVICE_FAMILY STRING "Cyclone"
// Retrieval info: PRIVATE: LPM_PIPELINE NUMERIC "0"
// Retrieval info: PRIVATE: Latency NUMERIC "0"
// Retrieval info: PRIVATE: OptionalSum NUMERIC "0"
// Retrieval info: PRIVATE: SYNTH_WRAPPER_GEN_POSTFIX STRING "0"
// Retrieval info: PRIVATE: SignedMult NUMERIC "1"
// Retrieval info: PRIVATE: USE_MULT NUMERIC "1"
// Retrieval info: PRIVATE: ValidConstant NUMERIC "1"
// Retrieval info: PRIVATE: WidthA NUMERIC "16"
// Retrieval info: PRIVATE: WidthB NUMERIC "5"
// Retrieval info: PRIVATE: WidthP NUMERIC "21"
// Retrieval info: PRIVATE: WidthS NUMERIC "1"
// Retrieval info: PRIVATE: aclr NUMERIC "0"
// Retrieval info: PRIVATE: clken NUMERIC "0"
// Retrieval info: PRIVATE: optimize NUMERIC "1"
// Retrieval info: CONSTANT: LPM_HINT STRING "INPUT_B_IS_CONSTANT=YES,MAXIMIZE_SPEED=5"
// Retrieval info: CONSTANT: LPM_REPRESENTATION STRING "SIGNED"
// Retrieval info: CONSTANT: LPM_TYPE STRING "LPM_MULT"
// Retrieval info: CONSTANT: LPM_WIDTHA NUMERIC "16"
// Retrieval info: CONSTANT: LPM_WIDTHB NUMERIC "5"
// Retrieval info: CONSTANT: LPM_WIDTHP NUMERIC "21"
// Retrieval info: USED_PORT: dataa 0 0 16 0 INPUT NODEFVAL dataa[15..0]
// Retrieval info: USED_PORT: result 0 0 21 0 OUTPUT NODEFVAL result[20..0]
// Retrieval info: CONNECT: @dataa 0 0 16 0 dataa 0 0 16 0
// Retrieval info: CONNECT: result 0 0 21 0 @result 0 0 21 0
// Retrieval info: CONNECT: @datab 0 0 5 0 10 0 0 0 0
// Retrieval info: LIBRARY: lpm lpm.lpm_components.all
// Retrieval info: GEN_FILE: TYPE_NORMAL mymul.v TRUE
// Retrieval info: GEN_FILE: TYPE_NORMAL mymul.inc FALSE
// Retrieval info: GEN_FILE: TYPE_NORMAL mymul.cmp FALSE
// Retrieval info: GEN_FILE: TYPE_NORMAL mymul.bsf FALSE
// Retrieval info: GEN_FILE: TYPE_NORMAL mymul_inst.v FALSE
// Retrieval info: GEN_FILE: TYPE_NORMAL mymul_bb.v TRUE
// Retrieval info: GEN_FILE: TYPE_NORMAL mymul_waveforms.html FALSE
// Retrieval info: GEN_FILE: TYPE_NORMAL mymul_wave*.jpg FALSE
// Retrieval info: LIB_FILE: lpm
