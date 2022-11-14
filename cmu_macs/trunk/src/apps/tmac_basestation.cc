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

class tmac_basestation : public mb_mblock
{
  mb_port_sptr 	d_tx;
  mb_port_sptr  d_rx;
  mb_port_sptr 	d_cs;
  pmt_t		d_tx_chan;	// returned tx channel handle

  enum state_t {
    INIT,
    IN_SYNC,
  };

  state_t	d_state;

  long d_local_addr;

 public:
  tmac_basestation(mb_runtime *runtime, const std::string &instance_name, pmt_t user_arg);
  ~tmac_basestation();
  void handle_message(mb_message_sptr msg);

 protected:
};

tmac_basestation::tmac_basestation(mb_runtime *runtime, const std::string &instance_name, pmt_t user_arg)
  : mb_mblock(runtime, instance_name, user_arg),
    d_state(INIT)
{ 
  // Extract USRP information from python and the arguments to the python script
  std::string mac("tmac");
  d_local_addr = 0;
  long total_nodes = pmt_to_long(pmt_nth(0, user_arg));
  long guard_time = pmt_to_long(pmt_nth(1, user_arg));

  pmt_t mac_data = pmt_list3(pmt_from_long(d_local_addr), pmt_from_long(total_nodes),pmt_from_long(guard_time));
  
  define_component(mac, mac, mac_data);
  d_tx = define_port("tx0", mac+"-tx", false, mb_port::INTERNAL);
  d_rx = define_port("rx0", mac+"-rx", false, mb_port::INTERNAL);
  d_cs = define_port("cs", mac+"-cs", false, mb_port::INTERNAL);

  connect("self", "tx0", mac, "tx0");
  connect("self", "rx0", mac, "rx0");
  connect("self", "cs", mac, "cs");
  
  std::cout << "[TMAC_BASESTATION] Initialized ..."
            << "\n    MAC: " << mac
            << "\n    Address: " << d_local_addr
            << "\n";

}

tmac_basestation::~tmac_basestation()
{
}

void
tmac_basestation::handle_message(mb_message_sptr msg)
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
        d_state = IN_SYNC;
        if(verbose)
          std::cout << "[TMAC_BASESTATION] In SYNC\n";
      }
      goto unhandled;


    //----------------------------- IN_SYNC -------------------------------//
    case IN_SYNC:
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
    std::cout << "[TMAC_BASESTATION] unhandled msg: " << msg
              << "in state "<< d_state << std::endl;
}

int
main (int argc, char **argv)
{
  mb_runtime_sptr rt = mb_make_runtime();
  pmt_t result = PMT_NIL;

  if(argc!=3) {
    std::cout << "usage: ./tmac_basestation total_nodes guard_time\n";
    return -1;
  }

  pmt_t args = pmt_list2(pmt_from_long(strtol(argv[1],NULL,10)),pmt_from_long(strtol(argv[2],NULL,10)));

  rt->run("top", "tmac_basestation", args, &result);
}

REGISTER_MBLOCK_CLASS(tmac_basestation);
