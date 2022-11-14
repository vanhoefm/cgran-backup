/* -*- c++ -*- */
/*
 * Copyright 2007 Free Software Foundation, Inc.
 * 
 * This file is part of GNU Radio
 * 
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 * 
 * GNU Radio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <mblock/mblock.h>
#include <mblock/runtime.h>
#include <mblock/protocol_class.h>
#include <mblock/exception.h>
#include <mblock/msg_queue.h>
#include <mblock/message.h>
#include <mblock/msg_accepter.h>
#include <mblock/class_registry.h>
#include <pmt.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>

#include <mac_symbols.h>

static bool verbose = false;

int timeval_subtract (struct timeval *result, struct timeval *x, struct timeval *y);

class tx_file : public mb_mblock
{
  mb_port_sptr 	d_tx;
  mb_port_sptr  d_rx;
  mb_port_sptr 	d_cs;
  pmt_t		d_tx_chan;	// returned tx channel handle

  enum state_t {
    INIT,
    TRANSMITTING,
  };

  state_t	d_state;
  long		d_nframes_xmitted;
  long d_tbytes;
  bool		d_done_sending;
  long    d_mac_max_payload;


  std::ifstream d_ifile;

  long d_local_addr;
  long d_dst_addr;

  pmt_t d_mac_properties;

  struct timeval d_start, d_end;

 public:
  tx_file(mb_runtime *runtime, const std::string &instance_name, pmt_t user_arg);
  ~tx_file();
  void handle_message(mb_message_sptr msg);

 protected:
  void open_usrp();
  void close_usrp();
  void allocate_channel();
  void enter_transmitting();
  void build_and_send_next_frame();
  void handle_xmit_response(pmt_t invocation_handle);
  void enter_closing_channel();
};

tx_file::tx_file(mb_runtime *runtime, const std::string &instance_name, pmt_t user_arg)
  : mb_mblock(runtime, instance_name, user_arg),
    d_state(INIT), 
    d_nframes_xmitted(0),
    d_tbytes(0),
    d_done_sending(false)
{ 
  // Extract USRP information from python and the arguments to the python script
  std::string mac = pmt_symbol_to_string(pmt_nth(0, user_arg));
  std::string file = pmt_symbol_to_string(pmt_nth(1, user_arg));
  d_local_addr = pmt_to_long(pmt_nth(2, user_arg));
  d_dst_addr = pmt_to_long(pmt_nth(3, user_arg));

  // Open a stream to the input file and ensure it's open
  d_ifile.open(file.c_str(), std::ios::binary|std::ios::in);

  if(!d_ifile.is_open()) {
    std::cout << "Error opening input file\n";
    shutdown_all(PMT_F);
    return;
  }

  pmt_t mac_data = pmt_list1(pmt_from_long(d_local_addr));
  
  define_component(mac, mac, mac_data);
  d_tx = define_port("tx0", mac+"-tx", false, mb_port::INTERNAL);
  d_rx = define_port("rx0", mac+"-rx", false, mb_port::INTERNAL);
  d_cs = define_port("cs", mac+"-cs", false, mb_port::INTERNAL);

  connect("self", "tx0", mac, "tx0");
  connect("self", "rx0", mac, "rx0");
  connect("self", "cs", mac, "cs");
  
  std::cout << "[TX_FILE] Initialized ..."
            << "\n    MAC: " << mac
            << "\n    Filename: " << file
            << "\n    Address: " << d_local_addr
            << "\n    Destination:" << d_dst_addr
            << "\n";

}

tx_file::~tx_file()
{

  d_ifile.close();
}

void
tx_file::handle_message(mb_message_sptr msg)
{
  pmt_t event = msg->signal();
  pmt_t data = msg->data();
  pmt_t port_id = msg->port_id();

  pmt_t handle = PMT_F;
  pmt_t status = PMT_F;
  pmt_t dict = PMT_NIL;
  std::string error_msg;

  // Dispatch based on state
  switch(d_state) {
    
    //------------------------------ INIT ---------------------------------//
    // When MAC is done initializing, it will send a response
    case INIT:
      
      if(pmt_eq(event, s_response_mac_initialized)) {
        handle = pmt_nth(0, data);
        status = pmt_nth(1, data);
        d_mac_properties = pmt_nth(2, data);

        if(pmt_is_dict(d_mac_properties)) {
          if(pmt_t mac_max_payload = pmt_dict_ref(d_mac_properties,
                                                  pmt_intern("max-payload"),
                                                  PMT_NIL)) {
            if(pmt_eqv(mac_max_payload, PMT_NIL)) {
              std::cout << "Error: MAC needs to send max payload with init message\n";
              shutdown_all(PMT_F);
            } else {
              d_mac_max_payload = pmt_to_long(mac_max_payload);
            }
          }
        } else {
          std::cout << "Error: MAC needs to send mac properties\n";
          shutdown_all(PMT_F);
        }

        // Set start time to keep track of performance
        gettimeofday(&d_start, NULL);

        if(pmt_eq(status, PMT_T)) {
          enter_transmitting();
          return;
        }
        else {
          error_msg = "error initializing mac:";
          goto bail;
        }
      }
      goto unhandled;

    //-------------------------- TRANSMITTING ----------------------------//
    // In the transmit state we count the number of underruns received and
    // ballpark the number with an expected count (something >1 for starters)
    case TRANSMITTING:
      
      // Check that the transmits are OK
      if (pmt_eq(event, s_response_tx_data)){
        handle = pmt_nth(0, data);
        status = pmt_nth(1, data);

        if (pmt_eq(status, PMT_T)){
          handle_xmit_response(handle);
          return;
        }
        else {
          error_msg = "bad response-tx-pkt:";
          goto bail;
        }
      }
      goto unhandled;

  }

 // An error occured, print it, and shutdown all m-blocks
 bail:
  std::cerr << error_msg << data
      	    << "status = " << status << std::endl;
  shutdown_all(PMT_F);
  return;

 // Received an unhandled message for a specific state
 unhandled:
  if(verbose && !pmt_eq(event, pmt_intern("%shutdown")))
    std::cout << "[TX_FILE] unhandled msg: " << msg
              << "in state "<< d_state << std::endl;
}

void
tx_file::enter_transmitting()
{
  d_state = TRANSMITTING;

  build_and_send_next_frame();
  build_and_send_next_frame();
  build_and_send_next_frame();
  build_and_send_next_frame();
}

void
tx_file::build_and_send_next_frame()
{
  size_t ignore;
  long n_bytes;

  if(d_done_sending)
    return;

  // Let's read in as much as possible to fit in a frame
  char data[d_mac_max_payload];
  d_ifile.read((char *)&data[0], sizeof(data));

  // Use gcount() and test if end of stream was met
  if(!d_ifile) {
    n_bytes = d_ifile.gcount();
    d_done_sending = true;
  } else {
    n_bytes = sizeof(data);
  }

  d_tbytes+=n_bytes;

  // Make the PMT data, get a writable pointer to it, then copy our data in
  pmt_t uvec = pmt_make_u8vector(n_bytes, 0);
  char *vdata = (char *) pmt_u8vector_writable_elements(uvec, ignore);
  memcpy(vdata, data, n_bytes);

  //  Transmit the data
  d_tx->send(s_cmd_tx_data,
	     pmt_list4(pmt_from_long(d_nframes_xmitted),   // invocation-handle
           pmt_from_long(d_dst_addr),// destination
		       uvec,				    // the samples
           PMT_NIL)); // per pkt properties

  d_nframes_xmitted++;

  if(verbose)
    std::cout << "[TX_FILE] Transmitted frame from " << d_local_addr
              << " to " << d_dst_addr 
              << " of size " << n_bytes << " bytes\n";

  std::cout << ".";
  fflush(stdout);
}


void
tx_file::handle_xmit_response(pmt_t handle)
{
  if (d_done_sending && pmt_to_long(handle)==(d_nframes_xmitted-1)){
    gettimeofday(&d_end, NULL);
    struct timeval result;
    timeval_subtract(&result, &d_end, &d_start);
    std::cout << "\n\nTransfer time: " 
              << result.tv_sec
              << "."
              << result.tv_usec
              << std::endl;

    float round_time=0;
    if(!pmt_eqv(PMT_NIL, pmt_dict_ref(d_mac_properties, pmt_intern("round-time"), PMT_NIL))) {
      round_time = pmt_to_long(pmt_dict_ref(d_mac_properties, pmt_intern("round-time"), PMT_NIL));
      round_time = round_time/64e6;
    }

      
    std::cout << "\n Round time: " << round_time << std::endl;

    float total_time = result.tv_sec + (result.tv_usec/(float)1000000) - round_time;

    std::cout << "\n\nTime: " << total_time << std::endl;
    std::cout << "\n\nThroughput: " << (d_tbytes*8/total_time/1000) << std::endl;
    fflush(stdout);
    shutdown_all(PMT_T);
  }

  // CMAC has taken care of waiting for the ACK
  build_and_send_next_frame();  
}

int
timeval_subtract (struct timeval *result, struct timeval *x, struct timeval *y)
{
  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_usec < y->tv_usec) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
    y->tv_usec -= 1000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_usec - y->tv_usec > 1000000) {
    int nsec = (x->tv_usec - y->tv_usec) / 1000000;
    y->tv_usec += 1000000 * nsec;
    y->tv_sec -= nsec;
  }

  /* Compute the time remaining to wait.
     tv_usec is certainly positive. */
  result->tv_sec = x->tv_sec - y->tv_sec;
  result->tv_usec = x->tv_usec - y->tv_usec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}

int
main (int argc, char **argv)
{
  mb_runtime_sptr rt = mb_make_runtime();
  pmt_t result = PMT_NIL;

  if(argc!=5) {
    std::cout << "usage: ./tx_file macs input_file local_addr dst_addr\n";
    std::cout << "  available macs: cmac\n";
    return -1;
  }

  pmt_t args = pmt_list4(pmt_intern(argv[1]), pmt_intern(argv[2]), pmt_from_long(strtol(argv[3],NULL,10)), pmt_from_long(strtol(argv[4],NULL,10)));

  rt->run("top", "tx_file", args, &result);
}

REGISTER_MBLOCK_CLASS(tx_file);
