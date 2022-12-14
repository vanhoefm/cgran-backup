// megafunction wizard: %ALTMULT_ADD%VBB%
// GENERATION: STANDARD
// VERSION: WM1.0
// MODULE: ALTMULT_ADD 

// ============================================================
// File Name: mult_add.v
// Megafunction Name(s):
// 			ALTMULT_ADD
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

module mult_add (
	clock0,
	dataa_0,
	dataa_1,
	dataa_2,
	dataa_3,
	datab_0,
	datab_1,
	datab_2,
	datab_3,
	ena0,
	result);

	input	  clock0;
	input	[15:0]  dataa_0;
	input	[15:0]  dataa_1;
	input	[15:0]  dataa_2;
	input	[15:0]  dataa_3;
	input	[5:0]  datab_0;
	input	[5:0]  datab_1;
	input	[5:0]  datab_2;
	input	[5:0]  datab_3;
	input	  ena0;
	output	[23:0]  result;

endmodule

// ============================================================
// CNX file retrieval info
// ============================================================
// Retrieval info: PRIVATE: ADDNSUB1_ACLR_SRC NUMERIC "3"
// Retrieval info: PRIVATE: ADDNSUB1_CLK_SRC NUMERIC "0"
// Retrieval info: PRIVATE: ADDNSUB1_PIPE_ACLR_SRC NUMERIC "3"
// Retrieval info: PRIVATE: ADDNSUB1_PIPE_CLK_SRC NUMERIC "0"
// Retrieval info: PRIVATE: ADDNSUB1_PIPE_REG STRING "1"
// Retrieval info: PRIVATE: ADDNSUB1_REG STRING "1"
// Retrieval info: PRIVATE: ADDNSUB3_ACLR_SRC NUMERIC "3"
// Retrieval info: PRIVATE: ADDNSUB3_CLK_SRC NUMERIC "0"
// Retrieval info: PRIVATE: ADDNSUB3_PIPE_ACLR_SRC NUMERIC "3"
// Retrieval info: PRIVATE: ADDNSUB3_PIPE_CLK_SRC NUMERIC "0"
// Retrieval info: PRIVATE: ADDNSUB3_PIPE_REG STRING "1"
// Retrieval info: PRIVATE: ADDNSUB3_REG STRING "1"
// Retrieval info: PRIVATE: ADD_ENABLE NUMERIC "1"
// Retrieval info: PRIVATE: ALL_REG_ACLR NUMERIC "0"
// Retrieval info: PRIVATE: A_ACLR_SRC_MULT0 NUMERIC "3"
// Retrieval info: PRIVATE: A_CLK_SRC_MULT0 NUMERIC "0"
// Retrieval info: PRIVATE: B_ACLR_SRC_MULT0 NUMERIC "3"
// Retrieval info: PRIVATE: B_CLK_SRC_MULT0 NUMERIC "0"
// Retrieval info: PRIVATE: HAS_MAC STRING "0"
// Retrieval info: PRIVATE: HAS_SAT_ROUND STRING "0"
// Retrieval info: PRIVATE: IMPL_STYLE_DEDICATED NUMERIC "0"
// Retrieval info: PRIVATE: IMPL_STYLE_DEFAULT NUMERIC "1"
// Retrieval info: PRIVATE: IMPL_STYLE_LCELL NUMERIC "0"
// Retrieval info: PRIVATE: INTENDED_DEVICE_FAMILY STRING "Cyclone"
// Retrieval info: PRIVATE: MULT_REGA0 NUMERIC "0"
// Retrieval info: PRIVATE: MULT_REGB0 NUMERIC "0"
// Retrieval info: PRIVATE: MULT_REGOUT0 NUMERIC "0"
// Retrieval info: PRIVATE: NUM_MULT STRING "4"
// Retrieval info: PRIVATE: OP1 STRING "Add"
// Retrieval info: PRIVATE: OP3 STRING "Add"
// Retrieval info: PRIVATE: OUTPUT_EXTRA_LAT NUMERIC "0"
// Retrieval info: PRIVATE: OUTPUT_REG_ACLR_SRC NUMERIC "3"
// Retrieval info: PRIVATE: OUTPUT_REG_CLK_SRC NUMERIC "0"
// Retrieval info: PRIVATE: Q_ACLR_SRC_MULT0 NUMERIC "3"
// Retrieval info: PRIVATE: Q_CLK_SRC_MULT0 NUMERIC "0"
// Retrieval info: PRIVATE: REG_OUT NUMERIC "1"
// Retrieval info: PRIVATE: RNFORMAT STRING "24"
// Retrieval info: PRIVATE: RQFORMAT STRING "Q1.15"
// Retrieval info: PRIVATE: RTS_WIDTH STRING "24"
// Retrieval info: PRIVATE: SAME_CONFIG NUMERIC "1"
// Retrieval info: PRIVATE: SAME_CONTROL_SRC_A0 NUMERIC "1"
// Retrieval info: PRIVATE: SAME_CONTROL_SRC_B0 NUMERIC "1"
// Retrieval info: PRIVATE: SCANOUTA NUMERIC "0"
// Retrieval info: PRIVATE: SCANOUTB NUMERIC "0"
// Retrieval info: PRIVATE: SHIFTOUTA_ACLR_SRC NUMERIC "3"
// Retrieval info: PRIVATE: SHIFTOUTA_CLK_SRC NUMERIC "0"
// Retrieval info: PRIVATE: SHIFTOUTA_REG STRING "0"
// Retrieval info: PRIVATE: SIGNA STRING "Signed"
// Retrieval info: PRIVATE: SIGNA_ACLR_SRC NUMERIC "3"
// Retrieval info: PRIVATE: SIGNA_CLK_SRC NUMERIC "0"
// Retrieval info: PRIVATE: SIGNA_PIPE_ACLR_SRC NUMERIC "3"
// Retrieval info: PRIVATE: SIGNA_PIPE_CLK_SRC NUMERIC "0"
// Retrieval info: PRIVATE: SIGNA_PIPE_REG STRING "1"
// Retrieval info: PRIVATE: SIGNA_REG STRING "1"
// Retrieval info: PRIVATE: SIGNB STRING "Signed"
// Retrieval info: PRIVATE: SIGNB_ACLR_SRC NUMERIC "3"
// Retrieval info: PRIVATE: SIGNB_CLK_SRC NUMERIC "0"
// Retrieval info: PRIVATE: SIGNB_PIPE_ACLR_SRC NUMERIC "3"
// Retrieval info: PRIVATE: SIGNB_PIPE_CLK_SRC NUMERIC "0"
// Retrieval info: PRIVATE: SIGNB_PIPE_REG STRING "1"
// Retrieval info: PRIVATE: SIGNB_REG STRING "1"
// Retrieval info: PRIVATE: SRCA0 STRING "Multiplier input"
// Retrieval info: PRIVATE: SRCB0 STRING "Multiplier input"
// Retrieval info: PRIVATE: SYNTH_WRAPPER_GEN_POSTFIX STRING "0"
// Retrieval info: PRIVATE: WIDTHA STRING "16"
// Retrieval info: PRIVATE: WIDTHB STRING "6"
// Retrieval info: LIBRARY: altera_mf altera_mf.altera_mf_components.all
// Retrieval info: CONSTANT: ADDNSUB_MULTIPLIER_ACLR1 STRING "UNUSED"
// Retrieval info: CONSTANT: ADDNSUB_MULTIPLIER_ACLR3 STRING "UNUSED"
// Retrieval info: CONSTANT: ADDNSUB_MULTIPLIER_PIPELINE_ACLR1 STRING "UNUSED"
// Retrieval info: CONSTANT: ADDNSUB_MULTIPLIER_PIPELINE_ACLR3 STRING "UNUSED"
// Retrieval info: CONSTANT: ADDNSUB_MULTIPLIER_PIPELINE_REGISTER1 STRING "CLOCK0"
// Retrieval info: CONSTANT: ADDNSUB_MULTIPLIER_PIPELINE_REGISTER3 STRING "CLOCK0"
// Retrieval info: CONSTANT: ADDNSUB_MULTIPLIER_REGISTER1 STRING "CLOCK0"
// Retrieval info: CONSTANT: ADDNSUB_MULTIPLIER_REGISTER3 STRING "CLOCK0"
// Retrieval info: CONSTANT: DEDICATED_MULTIPLIER_CIRCUITRY STRING "AUTO"
// Retrieval info: CONSTANT: INPUT_REGISTER_A0 STRING "UNREGISTERED"
// Retrieval info: CONSTANT: INPUT_REGISTER_A1 STRING "UNREGISTERED"
// Retrieval info: CONSTANT: INPUT_REGISTER_A2 STRING "UNREGISTERED"
// Retrieval info: CONSTANT: INPUT_REGISTER_A3 STRING "UNREGISTERED"
// Retrieval info: CONSTANT: INPUT_REGISTER_B0 STRING "UNREGISTERED"
// Retrieval info: CONSTANT: INPUT_REGISTER_B1 STRING "UNREGISTERED"
// Retrieval info: CONSTANT: INPUT_REGISTER_B2 STRING "UNREGISTERED"
// Retrieval info: CONSTANT: INPUT_REGISTER_B3 STRING "UNREGISTERED"
// Retrieval info: CONSTANT: INPUT_SOURCE_A0 STRING "DATAA"
// Retrieval info: CONSTANT: INPUT_SOURCE_A1 STRING "DATAA"
// Retrieval info: CONSTANT: INPUT_SOURCE_A2 STRING "DATAA"
// Retrieval info: CONSTANT: INPUT_SOURCE_A3 STRING "DATAA"
// Retrieval info: CONSTANT: INPUT_SOURCE_B0 STRING "DATAB"
// Retrieval info: CONSTANT: INPUT_SOURCE_B1 STRING "DATAB"
// Retrieval info: CONSTANT: INPUT_SOURCE_B2 STRING "DATAB"
// Retrieval info: CONSTANT: INPUT_SOURCE_B3 STRING "DATAB"
// Retrieval info: CONSTANT: INTENDED_DEVICE_FAMILY STRING "Cyclone"
// Retrieval info: CONSTANT: LPM_TYPE STRING "altmult_add"
// Retrieval info: CONSTANT: MULTIPLIER1_DIRECTION STRING "ADD"
// Retrieval info: CONSTANT: MULTIPLIER3_DIRECTION STRING "ADD"
// Retrieval info: CONSTANT: MULTIPLIER_REGISTER0 STRING "UNREGISTERED"
// Retrieval info: CONSTANT: MULTIPLIER_REGISTER1 STRING "UNREGISTERED"
// Retrieval info: CONSTANT: MULTIPLIER_REGISTER2 STRING "UNREGISTERED"
// Retrieval info: CONSTANT: MULTIPLIER_REGISTER3 STRING "UNREGISTERED"
// Retrieval info: CONSTANT: NUMBER_OF_MULTIPLIERS NUMERIC "4"
// Retrieval info: CONSTANT: OUTPUT_ACLR STRING "UNUSED"
// Retrieval info: CONSTANT: OUTPUT_REGISTER STRING "CLOCK0"
// Retrieval info: CONSTANT: PORT_ADDNSUB1 STRING "PORT_UNUSED"
// Retrieval info: CONSTANT: PORT_ADDNSUB3 STRING "PORT_UNUSED"
// Retrieval info: CONSTANT: PORT_SIGNA STRING "PORT_UNUSED"
// Retrieval info: CONSTANT: PORT_SIGNB STRING "PORT_UNUSED"
// Retrieval info: CONSTANT: REPRESENTATION_A STRING "SIGNED"
// Retrieval info: CONSTANT: REPRESENTATION_B STRING "SIGNED"
// Retrieval info: CONSTANT: SIGNED_ACLR_A STRING "UNUSED"
// Retrieval info: CONSTANT: SIGNED_ACLR_B STRING "UNUSED"
// Retrieval info: CONSTANT: SIGNED_PIPELINE_ACLR_A STRING "UNUSED"
// Retrieval info: CONSTANT: SIGNED_PIPELINE_ACLR_B STRING "UNUSED"
// Retrieval info: CONSTANT: SIGNED_PIPELINE_REGISTER_A STRING "CLOCK0"
// Retrieval info: CONSTANT: SIGNED_PIPELINE_REGISTER_B STRING "CLOCK0"
// Retrieval info: CONSTANT: SIGNED_REGISTER_A STRING "CLOCK0"
// Retrieval info: CONSTANT: SIGNED_REGISTER_B STRING "CLOCK0"
// Retrieval info: CONSTANT: WIDTH_A NUMERIC "16"
// Retrieval info: CONSTANT: WIDTH_B NUMERIC "6"
// Retrieval info: CONSTANT: WIDTH_RESULT NUMERIC "24"
// Retrieval info: USED_PORT: clock0 0 0 0 0 INPUT VCC "clock0"
// Retrieval info: USED_PORT: dataa_0 0 0 16 0 INPUT GND "dataa_0[15..0]"
// Retrieval info: USED_PORT: dataa_1 0 0 16 0 INPUT GND "dataa_1[15..0]"
// Retrieval info: USED_PORT: dataa_2 0 0 16 0 INPUT GND "dataa_2[15..0]"
// Retrieval info: USED_PORT: dataa_3 0 0 16 0 INPUT GND "dataa_3[15..0]"
// Retrieval info: USED_PORT: datab_0 0 0 6 0 INPUT GND "datab_0[5..0]"
// Retrieval info: USED_PORT: datab_1 0 0 6 0 INPUT GND "datab_1[5..0]"
// Retrieval info: USED_PORT: datab_2 0 0 6 0 INPUT GND "datab_2[5..0]"
// Retrieval info: USED_PORT: datab_3 0 0 6 0 INPUT GND "datab_3[5..0]"
// Retrieval info: USED_PORT: ena0 0 0 0 0 INPUT VCC "ena0"
// Retrieval info: USED_PORT: result 0 0 24 0 OUTPUT GND "result[23..0]"
// Retrieval info: CONNECT: @datab 0 0 6 6 datab_1 0 0 6 0
// Retrieval info: CONNECT: @datab 0 0 6 12 datab_2 0 0 6 0
// Retrieval info: CONNECT: @datab 0 0 6 18 datab_3 0 0 6 0
// Retrieval info: CONNECT: @clock0 0 0 0 0 clock0 0 0 0 0
// Retrieval info: CONNECT: result 0 0 24 0 @result 0 0 24 0
// Retrieval info: CONNECT: @dataa 0 0 16 0 dataa_0 0 0 16 0
// Retrieval info: CONNECT: @dataa 0 0 16 16 dataa_1 0 0 16 0
// Retrieval info: CONNECT: @dataa 0 0 16 32 dataa_2 0 0 16 0
// Retrieval info: CONNECT: @ena0 0 0 0 0 ena0 0 0 0 0
// Retrieval info: CONNECT: @dataa 0 0 16 48 dataa_3 0 0 16 0
// Retrieval info: CONNECT: @datab 0 0 6 0 datab_0 0 0 6 0
// Retrieval info: GEN_FILE: TYPE_NORMAL mult_add.v TRUE FALSE
// Retrieval info: GEN_FILE: TYPE_NORMAL mult_add.inc FALSE FALSE
// Retrieval info: GEN_FILE: TYPE_NORMAL mult_add.cmp FALSE FALSE
// Retrieval info: GEN_FILE: TYPE_NORMAL mult_add.bsf FALSE FALSE
// Retrieval info: GEN_FILE: TYPE_NORMAL mult_add_inst.v FALSE FALSE
// Retrieval info: GEN_FILE: TYPE_NORMAL mult_add_bb.v TRUE FALSE
// Retrieval info: GEN_FILE: TYPE_NORMAL mult_add_waveforms.html TRUE FALSE
// Retrieval info: GEN_FILE: TYPE_NORMAL mult_add_wave*.jpg FALSE FALSE
// Retrieval info: LIB_FILE: altera_mf
