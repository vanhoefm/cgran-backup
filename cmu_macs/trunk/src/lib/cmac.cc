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

#include <cmac.h>

static bool verbose = false;

static pmt_t s_timeout = pmt_intern("%timeout");

cmac::cmac(mb_runtime *rt, const std::string &instance_name, pmt_t user_arg)
  : mac(rt, instance_name, user_arg),
  d_state(INIT_CMAC),
  d_framer_state(SYNC_SEARCH),
  d_verbose_frames(true),
  d_nframes_recvd(0)
{
  define_mac_ports();   // Initialize ports for message passing
  d_local_address = pmt_to_long(pmt_nth(0, user_arg));
}

cmac::~cmac()
{
}

void cmac::define_mac_ports()
{

  // Define ports for the application to connect to us
  d_tx = define_port("tx0", "cmac-tx", true, mb_port::EXTERNAL);  // Transmit
  d_rx = define_port("rx0", "cmac-rx", true, mb_port::EXTERNAL);  // Receive
  d_cs = define_port("cs", "cmac-cs", true, mb_port::EXTERNAL);   // Control/Status
}

// Invoked when the base 'mac' class finishes initializing the USRP
void cmac::usrp_initialized()
{
  initialize_cmac();
}

void cmac::initialize_cmac()
{
  set_carrier_sense(false, 25, 0, PMT_NIL);   // Initial carrier sense setting

  d_state = IDLE;   // State where we wait for messages to do something

  d_seq_num = 0;    // Sequence number frames

  // Send any information between the MAC and the PHY, require max frame sizes
  pmt_t mac_properties = pmt_make_dict();
  pmt_dict_set(mac_properties, pmt_intern("max-frame"), pmt_from_long(MAX_FRAME_SIZE));
  pmt_dict_set(mac_properties, pmt_intern("max-payload"), pmt_from_long(max_frame_payload()));
  
  d_cs->send(s_response_mac_initialized,   // Notify the application that MAC 
             pmt_list3(PMT_NIL, PMT_T,mac_properties)); // hs been initialized

  enable_rx();

  std::cout << "[CMAC] Initialized, and idle\n";
}

// This is the crux of the MAC layer.  The general architecture is to have
// states at which the MAC is in, and then handle messages from the application
// and physical layer based on that state.  
// 
// Then, the incoming 'event' (type of message) in the state triggers some
// functionality.  For example, if we are in the idle state and receive a 
// s_cmd_tx_data event from the application (detected by the port ID), we send
// the data to the physical layer to be modulated.
void cmac::handle_mac_message(mb_message_sptr msg)
{
  pmt_t event = msg->signal();      // type of message
  pmt_t data = msg->data();         // the associated data
  pmt_t port_id = msg->port_id();   // the port the msg was received on

  std::string error_msg;

  switch(d_state) {
    
    //----------------------------- INIT CMAC --------------------------------//
    // In the INIT_CMAC state, now that the USRP is initialized we can do things
    // like right the carrier sense threshold to the FPGA register.
    case INIT_CMAC:
      goto unhandled;
    
    //----------------------------- IDLE ------------------------------------//
    // In the idle state the MAC is not quite 'idle', it is just not doing
    // anything specific.  It is still being passive with data between the
    // application and the lower layer.
    case IDLE:
      
      //---- Port: CMAC TX -------------- State: IDLE -----------------------//
      if(pmt_eq(d_tx->port_symbol(), port_id)) {

        if(pmt_eq(event, s_cmd_tx_data)) {
          build_frame(data); 
        }
        return;
      }
      
      //---- Port: CMAC CS -------------- State: IDLE -----------------------//
      if(pmt_eq(d_cs->port_symbol(), port_id)) {

        if(pmt_eq(event, s_cmd_rx_enable)) {
          enable_rx();                                // Enable RX
        }
        else if(pmt_eq(event, s_cmd_rx_disable)) {
          disable_rx();                               // Disable RX
        }
        return;
      }

      //---- Port: GMSK CS -------------- State: IDLE -----------------------//
      if(pmt_eq(d_phy_cs->port_symbol(), port_id)) {
        
        if(pmt_eq(event, s_response_demod)) {
          incoming_data(data);                       // Incoming frame!
        }
        return;
      }
      goto unhandled;


    //---------------------------- ACK WAIT ----------------------------------//
    // In this state, we are waiting for an ACK back before we pass up a
    // "successfull transmission" message to the application.  If we do not get
    // an ACK within a timeout, we retransmit.
    case ACK_WAIT:

      //--------- ACK TIMEOUT -----------------------------------------------//
      // FAIL! We've hit an ACK timeout, time to panic and retransmit.
      // The timer automatically resets, if we want to get smart we can keep
      // track of how many times its fired before we tell the application the
      // transmission has failed.
      if(pmt_eq(event, s_timeout)) {
        std::cout << "x";
        fflush(stdout);
        d_seq_num--;
        build_frame(d_last_frame);
        return;
      }
      
      //---- Port: GMSK CS -------------- State: ACK_WAIT -------------------//
      if(pmt_eq(d_phy_cs->port_symbol(), port_id)) {

        if(pmt_eq(event, s_response_demod)) {
          incoming_data(data);                       // Incoming data
        }
        return;
      }
      goto unhandled;

    
    //---------------------------- SEND ACK ----------------------------------//
    // Expect nothing in this state, we get a packet_transmitted() once we get
    // a response that the ACK has left the host.
    case SEND_ACK:
      goto unhandled;

  } // End of switch()
  
 // Received an unhandled message for a specific state
 unhandled:
  if(0 && verbose && !pmt_eq(event, pmt_intern("%shutdown")))
    std::cout << "[CMAC] unhandled msg: " << msg
              << "in state "<< d_state << std::endl;
}

// Method to handle setting carrier sense within the FPGA via register writes.
void cmac::set_carrier_sense(bool toggle, long threshold, long deadline, pmt_t invocation)
{
  d_carrier_sense = toggle;

  d_us_tx->send(s_cmd_to_control_channel, 
             pmt_list2(invocation, 
                       pmt_list1(
                            pmt_list2(s_op_write_reg, 
                                      pmt_list2(
                                      pmt_from_long(REG_CS_THRESH), 
                                      pmt_from_long(threshold))))));

  d_us_tx->send(s_cmd_to_control_channel, 
             pmt_list2(invocation, 
                       pmt_list1(
                            pmt_list2(s_op_write_reg, 
                                      pmt_list2(
                                      pmt_from_long(REG_CS_DEADLINE), 
                                      pmt_from_long(deadline))))));

  d_cs_thresh = threshold;    // Save our new threshold
  d_cs_deadline = deadline;   // Keep the deadline

  if(verbose)
    std::cout << "[CMAC] Setting carrier sense to " << toggle 
              << "\n ... threshold: " << d_cs_thresh
              << "\n ... deadline:  " << d_cs_deadline
              << std::endl;
}

// Invoked when we get a response that a packet was written to the USRP USB bus,
// we assume that it has been transmitted (or will be, within negligable time).
//
// The application is NOT informed that the packet was transmitted successfully
// until an ACK is received, so we enter an ACK wait state.
void cmac::packet_transmitted(pmt_t data)
{
  fflush(stdout);
  pmt_t invocation_handle = pmt_nth(0, data);
  pmt_t status = pmt_nth(1, data);

  enable_rx();  // Need to listen for ACKs

  // If we are already in the SEND_ACK state, we were waiting for an ACK to be
  // transmitted, once successful we assume it was received and go to IDLE
  if(d_state == SEND_ACK) {
    d_state = IDLE;
    return;
  }

  // If we were in the ACK_WAIT state, we don't need to schedule another timeout
  if(d_state == ACK_WAIT)
    return;
  
  // Schedule an ACK timeout to fire every timeout period. This should be user
  // settable.  The first timeout is now+TIMEOUT_PERIOD
  const double TIMEOUT_PERIOD = 0.5;  // in seconds
  mb_time now = mb_time::time();
  d_ack_timeout = schedule_periodic_timeout(now + TIMEOUT_PERIOD, mb_time(TIMEOUT_PERIOD), PMT_T);
  
  d_state = ACK_WAIT;

  if(verbose)
    std::cout << "[CMAC] Packet transmitted, going to ACK wait\n";
}

// An incoming frame from the physical layer for us!  We check the packet
// properties to determine the sender and if it passed a CRC check, for example.
void cmac::incoming_frame(pmt_t data)
{
  pmt_t invocation_handle = PMT_NIL;
  pmt_t payload = pmt_nth(0, data);
  pmt_t pkt_properties = pmt_nth(1, data);
  std::string status;

  // Properties are set in the physical layer framing code
  long src = pmt_to_long(pmt_dict_ref(pkt_properties, pmt_intern("src"), PMT_NIL));
  long dst = pmt_to_long(pmt_dict_ref(pkt_properties, pmt_intern("dst"), PMT_NIL));
  bool crc = pmt_to_bool(pmt_dict_ref(pkt_properties, pmt_intern("crc"), PMT_NIL));
  bool ack = pmt_to_bool(pmt_dict_ref(pkt_properties, pmt_intern("ack"), PMT_NIL));
  unsigned long seq = (unsigned long) pmt_to_long(pmt_dict_ref(pkt_properties, 
                                                               pmt_intern("seq"), 
                                                               PMT_NIL));

  if(ack) {         // Handle ACKs in a special manner
    handle_ack(src, dst);
    return;
  }

  if(dst != d_local_address)  // not for this address
    return;
  
  if(crc) {  // CRC passes, let's ACK the source
    build_and_send_ack(src);
    status="pass";
  } else {
    status="fail";
  }
  
  if(d_verbose_frames) {
    if(d_nframes_recvd==0)
      std::cout << "crc\trecvd\tseq#\tsrc\n";
    std::cout << status << "\t"
              << d_nframes_recvd << "\t"
              << seq << "\t"
              << src << "\t"
              << std::endl;
  }
  
  d_nframes_recvd++;

  d_rx->send(s_response_rx_pkt, pmt_list3(invocation_handle, payload, pkt_properties));
}

// Special handling for ACK frames
void cmac::handle_ack(long src, long dst)
{
  // CMAC does not care about frames if we're not in the ACK_WAIT state
  if(d_state!=ACK_WAIT)
    return;

  cancel_timeout(d_ack_timeout);    // Cancel our ACK timeout

  // Now that we have an ACK, we can notify the application of a successfully TX
  pmt_t invocation_handle = pmt_nth(0, d_last_frame);
  d_tx->send(s_response_tx_data,
             pmt_list2(invocation_handle,
                       PMT_T));

  disable_rx();     // FIXME: spend more time thinking about this, I think its incorrect

  d_state = IDLE;   // Back to the idle state!

  if(verbose)
    std::cout << "[CMAC] Got ACK, going back to idle\n";

  return;
}

void cmac::build_and_send_ack(long dst)
{
  size_t ignore;
  char data;
  long n_bytes=1;   // Negligable payload
  
  disable_rx();     // No need to receive while transmitting, not required,
                    // only saves processing power.

  // Make the PMT data, get a writable pointer to it, then copy our data in
  pmt_t uvec = pmt_make_u8vector(n_bytes, 0);
  char *vdata = (char *) pmt_u8vector_writable_elements(uvec, ignore);
  memcpy(vdata, &data, n_bytes);

  // Per packet properties
  pmt_t tx_properties = pmt_make_dict();
  pmt_dict_set(tx_properties, pmt_intern("ack"), PMT_T);  // it's an ACK!

  pmt_t pdata = pmt_list4(PMT_NIL,                        // No invocation.
                          pmt_from_long(dst),             // To them.
                          uvec,                           // With data.
                          tx_properties);                 // It's an ACK!

  build_frame(pdata);
  
  d_state = SEND_ACK;                   // Switch MAC states
  
  if(verbose)
    std::cout << "[CMAC] Transmitted ACK from " << d_local_address
              << " to " << dst
              << std::endl;
}

// Entrance of new incoming data
void cmac::incoming_data(pmt_t data)
{
  pmt_t bits = pmt_nth(0, data);
  pmt_t demod_properties = pmt_nth(1, data);
  std::vector<unsigned char> bit_data = boost::any_cast<std::vector<unsigned char> >(pmt_any_ref(bits));

  framer(bit_data, demod_properties);
}

REGISTER_MBLOCK_CLASS(cmac);
