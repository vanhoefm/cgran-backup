/* -*- c++ -*- */
/*
 * Copyright 2004,2006 Free Software Foundation, Inc.
 * 
 * This file is part of GNU Radio
 * 
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 * 
 * GNU Radio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <listener_reader_monitor_cmd_gate.h>
#include <gr_io_signature.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <float.h>
#include "reader_monitor_vars.h"



/////////////////////////////////////////////////////////////////
// INITIAL SETUP
////////////////////////////////////////////////////////////////
listener_reader_monitor_cmd_gate_sptr
listener_make_reader_monitor_cmd_gate (float us_per_sample, bool real_time)
{
  return listener_reader_monitor_cmd_gate_sptr (new listener_reader_monitor_cmd_gate (us_per_sample, real_time));
}

listener_reader_monitor_cmd_gate::listener_reader_monitor_cmd_gate(float us_per_sample, bool real_time)
  : gr_block("listener_reader_monitor_cmd_gate",
		  gr_make_io_signature (2,2,sizeof(float)),
		      gr_make_io_signature (1,1,sizeof(float))),
    d_us_per_sample(us_per_sample), d_real_time(real_time)
{
  
	d_delim_width = (int)((1 / us_per_sample) * 12.5);
	d_reader_freq = 0.;
	state = IDLE;
	pwr_dwn_cnt = 0;
	log_q = gr_make_msg_queue(100000);
	pwrd_dwn = true;
	frame_sync=false;
	tag = false;
	IS_TAG=false;
	rtcal = 10000; // reset later
	pwr_gate_cnt = 0;  
	
}
// END INITIAL SETUP
//////////////////////////////////////////////////////////////////////////////////////////////////


   
listener_reader_monitor_cmd_gate::~listener_reader_monitor_cmd_gate()
{
}




////////////////////////////////////////////////////////////////////////////////////////////////
// SET READER FREQUENCY
////////////////////////////////////////////////////////////////////////////////////////////////
void listener_reader_monitor_cmd_gate::set_reader_freq (float new_reader_freq)
{
	d_reader_freq = new_reader_freq;
}
// END SET READER FREQUENCY
///////////////////////////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////////////////////////////
// GENERAL WORK
//////////////////////////////////////////////////////////////////////////////////////////////////
int listener_reader_monitor_cmd_gate::general_work(int noutput_items,
			   			gr_vector_int &ninput_items,
			   			gr_vector_const_void_star &input_items,
			   			gr_vector_void_star &output_items)
{
	const float * in = (const float *)input_items[0];
	const float * in_TAG = (const float *)input_items[1];
	float * out = (float * )output_items[0];

	double avg, max, min;
	double max_RSSI, avg_RSSI;
	static long high_cnt, low_cnt;
  
	int n_out = 0;
  
	if(state != DATA) {  
		max_min(in, noutput_items, &max, &min, &avg);
		max_RSSI = max;
		avg_RSSI = avg;
		d_thresh = max * .6;
	}
 
  	for (int i = 0; i < noutput_items; i++){
    		pwr_dwn_cnt += d_us_per_sample;
       		if(in[i] > d_thresh) {
      			if(low_cnt > 0){
        			int end_command = determine_symbol(high_cnt, low_cnt, d_bits, max_RSSI, avg_RSSI);
				low_cnt = 0;
				high_cnt = 0;
			}
      			high_cnt++;
      			if(pwrd_dwn==false && high_cnt * d_us_per_sample > rtcal && state == DATA) { // tag reply
				tag = true;
      			}
      			else {
				tag = false;
        			IS_TAG = false;
      			} // END TAG REPLY if
    		} // END if > d_thresh
    
    		if(in[i] < d_thresh && d_thresh > 5000){
     	 		low_cnt++;
               		if(state == DATA) {
				if(low_cnt * d_us_per_sample > d_delim_width * d_us_per_sample * 3) {  //Powered Down
	  				state = IDLE;
	  				decode_command(d_bits, d_len_bits, 0, max_RSSI, avg_RSSI, OKAY);
	  				gr_message_sptr log_msg =  gr_make_message(PWR_DWN, pwr_dwn_cnt - (high_cnt * d_us_per_sample), d_reader_freq, 0); 
	  				log_q->insert_tail(log_msg);
					pwr_dwn_cnt = 0;
					low_cnt = 0;
					high_cnt = 0;
					pwrd_dwn = true;
					tag = false;
					IS_TAG = false;
	  			}
      			} //END if STATE==DATA
    		} //END if < d_thresh

		if(tag==true && in[i] > d_thresh && d_thresh > 5000) {
      			IS_TAG = true;
      			out[n_out++] = in_TAG[i]; //Output samples
    		}

	} //END FOR

	consume_each(noutput_items);
	return n_out;
 
}
// END GENERAL WORK
////////////////////////////////////////////////////////////////////////////////////////// 



//////////////////////////////////////////////////////////////////////////////////////////
// DETERMINE SYMBOL
//////////////////////////////////////////////////////////////////////////////////////////
int listener_reader_monitor_cmd_gate::determine_symbol(int high_cnt, int low_cnt, char * bits, double max_RSSI, double avg_RSSI)
{

	//Return codes:
	// -1: error
	//  0: added bit
	//  1: end of command 

	if(state == IDLE) {
     		d_len_bits=0;
     		if(low_cnt > d_delim_width * .75 && low_cnt < d_delim_width * 1.25 && high_cnt * d_us_per_sample > 1000) { //NEW DELIMITER
      			state = DELIM;
      			//printf("power_UP duration = %f\n", high_cnt*d_us_per_sample);
      			cmd_len = low_cnt * d_us_per_sample;
     		}
  	}
  	else if(state == DELIM) { //DELIMITER FIELD
    		d_len_bits=0;
    		if((high_cnt + low_cnt) * d_us_per_sample < 5 || (high_cnt + low_cnt) * d_us_per_sample > 28) {   //Invalid TARI
      			state = IDLE;
      			//printf("Invalid TARI:%f\n", (high_cnt + low_cnt) * d_us_per_sample);
      			return -1;
    		}
    		tari = (high_cnt + low_cnt) * d_us_per_sample;
    		cmd_len += tari;
    		//printf("CORRECT TARI\n");
    		if(low_cnt * d_us_per_sample < 1 || low_cnt * d_us_per_sample < .265 * tari || low_cnt * d_us_per_sample > .525 * tari) { // Invalid PW
      			state = IDLE;
      			//printf("Invalid PW: %f TARI: %f\n", low_cnt * d_us_per_sample, tari);
      			return -1;
    		}
    		pw = low_cnt * d_us_per_sample;
    		state = PW; 
    		//printf("Tari: %f PW:%f\n", tari, pw);
	}
  	else if(state == PW) { //PW FIELD
    		d_len_bits=0;
    		if((high_cnt + low_cnt) * d_us_per_sample < 2.2 * tari || (high_cnt + low_cnt) * d_us_per_sample > 3.5 * tari) {  //Invalid RTCal
      			state = IDLE;
      			//printf("Invalid RTCal:%f TARI: %f  \n", (high_cnt + low_cnt) * d_us_per_sample, tari);
     			//gr_message_sptr log_msg =  gr_make_message(INVALID_RTCAL, 0, ERROR, 0); 
      			//log_q->insert_tail(log_msg);
      			return -1;
    		}
    		rtcal = (high_cnt + low_cnt) * d_us_per_sample;
    		//printf("CORRECT RTCal:%f TARI: %f  \n", rtcal, tari);
    		state = RTCAL;
    		cmd_len += rtcal;
  	}
	else if(state == RTCAL) { //RTCAL FIELD
    		d_len_bits=0;
    		if((high_cnt + low_cnt) * d_us_per_sample > 5 * rtcal) {  //Invalid TRCAL
      			state = IDLE;
      			//printf("Invalid TRCal/DATA:%f\n", (high_cnt + low_cnt) * d_us_per_sample);
      			//gr_message_sptr log_msg =  gr_make_message(INVALID_TRCAL, 0, ERROR, 0); 
      			//log_q->insert_tail(log_msg);
      			return -1;
    		}
  		if((high_cnt + low_cnt) * d_us_per_sample > rtcal) {  //CORRECT TRCal
      			trcal = (high_cnt + low_cnt) * d_us_per_sample; 
      			//printf("-----CORRECT TRcal=%f, RTcal=%f, Tari=%f \n",trcal, rtcal, tari); 
      			frame_sync = false;
      			d_len_bits=0;
    		}
    		else if((high_cnt + low_cnt) * d_us_per_sample > 1.4 * tari) {  //Framesync: TRCAL==BIT-1
      			bits[0] = '1';
      			d_len_bits=1;
      			frame_sync = true;
      			//printf("-----MUST BE FRAMESYNC bit 1 \n"); 
    		}
    		else {   //Framesync: TRCAL==BIT-0
      			bits[0] = '0';
      			d_len_bits=1;
      			frame_sync = true;
      			//printf("-----MUST BE FRAMESYNC bit 0\n"); 
    		}
		state = DATA;
   		cmd_len += (high_cnt + low_cnt) * d_us_per_sample;
	}
  	else if(state == DATA) { //DATA FIELD
    		mid_pt = d_thresh;
    		if((high_cnt + low_cnt) * d_us_per_sample > 2.5 * rtcal) {  //Maybe a new burst
      			inter_arrival = high_cnt * d_us_per_sample;
      			//if(low_cnt * d_us_per_sample > d_delim_width * d_us_per_sample * .90){
      			if(low_cnt > d_delim_width * .75 && low_cnt < d_delim_width * 1.25) { // new delimiter, decode previous message
				state = DELIM;
				tag = false;
        			IS_TAG = false;
				decode_command(bits, d_len_bits, inter_arrival, max_RSSI, avg_RSSI, OKAY);
				cmd_len = low_cnt * d_us_per_sample;
				//printf("Inter-arrival: %f\n", inter_arrival);
      			}
      			else {  //ERROR: it's not a new burst
				state = IDLE;
				tag = false;
        			IS_TAG = false;
        			decode_command(bits, d_len_bits, max_RSSI, avg_RSSI, inter_arrival, OKAY );
				return -1;
      			}
      			return 1;
		}
   		else if((high_cnt + low_cnt) * d_us_per_sample > 1.4 * tari) { //data-1
      			bits[d_len_bits] = '1';
      			d_len_bits++;
      			cmd_len += (high_cnt + low_cnt) * d_us_per_sample;
      		}
    		else {   //data-0
      			bits[d_len_bits] = '0';
      			d_len_bits++;
      			cmd_len += (high_cnt + low_cnt) * d_us_per_sample;
    		}
	}
  	
	return 0;

}
// END DETERMINE SYMBOL
//////////////////////////////////////////////////////////////////////////////////////////



//////////////////////////////////////////////////////////////////////////////////////////
// DECODE COMMAND
//////////////////////////////////////////////////////////////////////////////////////////
int listener_reader_monitor_cmd_gate::decode_command(char * bits, int len_bits, float inter_arrival, double max_RSSI, double avg_RSSI, int flag)
{

	bits[len_bits] = '\0';  
	char tmp_str[10000];
	timeval time;
	gettimeofday(&time, NULL);
	tm * t_info = gmtime(&time.tv_sec);
	int len = sprintf(tmp_str, "Tari:%f/PW:%f/RTCal:%f/TRCal:%f/CMD:%s/COMMAND_LEN:%f/READER_FREQ:%f/MAX_RSSI:%.2f/AVG_RSSI:%.2f/TIME:%d.%03ld",tari, pw, rtcal, trcal, bits, cmd_len,d_reader_freq,max_RSSI,avg_RSSI,(t_info->tm_hour*3600)+(t_info->tm_min*60)+t_info->tm_sec, time.tv_usec/1000);

	if(pwrd_dwn) {
    		gr_message_sptr log_msg =  gr_make_message(PWR_UP, pwr_dwn_cnt, d_reader_freq, 0); 
    		log_q->insert_tail(log_msg);
    		pwr_dwn_cnt = cmd_len + inter_arrival;
    		pwrd_dwn = false;
  	}
  
  	gr_message_sptr log_msg =  gr_make_message(READER_MSG, inter_arrival, flag, len); 
  
  	// REAL TIME PRINTF ENABLED, print also reader freq.
  	if (d_real_time==true) {
		if (bits[0]=='1' && bits[1]=='0' && bits[2]=='0' && bits[3]=='0') {
			printf("*****NEW READER MESSAGE  @ %.1f MHz: QUERY COMMAND******\n",d_reader_freq);
  		}
 		if (bits[0]=='0' && bits[1]=='0' && len_bits==4) {
			printf("*****NEW READER MESSAGE  @ %.1f MHz: QRep Session ******\n",d_reader_freq);
  		}
		if (bits[0]=='0' && bits[1]=='1' && len_bits==18) {
			printf("*****NEW READER MESSAGE @ %.1f MHz: ACK ***************\n",d_reader_freq);
			printf("%s\n",bits);
  		}
		if (bits[0]=='1' && bits[1]=='1' && bits[2]=='0' && bits[3]=='0') {
			printf("*****NEW READER MESSAGE @ %.1f MHz: NAK ***************\n",d_reader_freq);
  		}
		if (bits[0]=='1' && bits[1]=='0' && bits[2]=='0' && bits[3]=='1' && len_bits==9) {
			printf("*****NEW READER MESSAGE  @ %.1f MHz: QAdj Session ******\n",d_reader_freq);
  		}
		if (bits[0]=='1' && bits[1]=='1' && bits[2]=='0' && bits[3]=='0' && bits[4]=='0' && bits[5]=='0' && bits[6]=='0' && bits[7]=='0') {
			printf("*****NEW READER MESSAGE  @ %.1f MHz: NAK ***************\n",d_reader_freq);
  		}
		if (bits[0]=='1' && bits[1]=='0' && bits[2]=='1' && bits[3]=='0' ) {
			printf("*****NEW READER MESSAGE  @ %.1f MHz: SELECT ************\n",d_reader_freq);
  		}
	}
	
  	memcpy(log_msg->msg(), tmp_str, len);
  	log_q->insert_tail(log_msg);
  	d_len_bits = 0;
	
	return 1;

}
// END DECODE COMMAND
//////////////////////////////////////////////////////////////////////////////////////////  



//////////////////////////////////////////////////////////////////////////////////////////
// CALCULATE MAX AND MIN OF THE SIGNAL
//////////////////////////////////////////////////////////////////////////////////////////
int listener_reader_monitor_cmd_gate::max_min(const float * buffer, int len, double * max, double * min, double * avg )
{

	double tmp_avg = 0;
	double tmp_std_dev = 0;

	for (int i = 0; i < len; i++) {
    		tmp_avg += buffer[i];
    		if(buffer[i] > * max) {
      			*max = buffer[i];
    		}
    		if(buffer[i] < * min) {
      			*min = buffer[i];
    		}
  	}
  	tmp_avg = tmp_avg / len;
  	*avg = tmp_avg;
  
  	return 1;

}
// END CALCULATE MAX AND MIN OF THE SIGNAL
//////////////////////////////////////////////////////////////////////////////////////////



//////////////////////////////////////////////////////////////////////////////////////////
// FORECAST
//////////////////////////////////////////////////////////////////////////////////////////
void listener_reader_monitor_cmd_gate::forecast (int noutput_items, gr_vector_int &ninput_items_required)
{
	unsigned ninputs = ninput_items_required.size ();
	for (unsigned i = 0; i < ninputs; i++){
		ninput_items_required[i] = noutput_items;
	}   
}
// END FORECAST
//////////////////////////////////////////////////////////////////////////////////////////
