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

#include <tmac.h>

static int INITIAL_SYNC = 114967296;

static bool verbose = true;

static pmt_t s_timeout = pmt_intern("%timeout");

tmac::tmac(mb_runtime *rt, const std::string &instance_name, pmt_t user_arg)
  : mac(rt, instance_name, user_arg),
  d_state(INIT_TMAC),
  d_base_station(false),
  d_last_sync(0)
{
  define_mac_ports();   // Initialize ports for message passing
  
  // Make sure a local address was specified, convert it from PMT to long
  if(!pmt_eqv(pmt_nth(0, user_arg), PMT_NIL)) {
    d_local_address = pmt_to_long(pmt_nth(0, user_arg));
  } else {
    std::cout << "[TMAC] ERROR: Need to specify local address when initializing MAC\n";
    shutdown_all(PMT_F);
  }

  // The local address 0 is reserved for the base station, the total number of
  // nodes needs to be specified if so.
  if(d_local_address==0) {
    d_base_station=true;

    if(!pmt_eqv(pmt_nth(1, user_arg), PMT_NIL) && !pmt_eqv(pmt_nth(2, user_arg), PMT_NIL)) {
      d_total_nodes = pmt_to_long(pmt_nth(1, user_arg));
      d_guard_time = pmt_to_long(pmt_nth(2, user_arg));
      if(verbose)
        std::cout << "[TMAC] Initializing base station with " 
                  << d_total_nodes << " nodes  "
                  << d_guard_time << " guard time\n";

    } else {
      std::cout << "[TMAC] ERROR: Need to specify total number of nodes and guard time in the network\n";
      shutdown_all(PMT_F);
    }
  }
}

tmac::~tmac()
{
}

void tmac::define_mac_ports()
{
  // Define ports for the application to connect to us
  d_tx = define_port("tx0", "tmac-tx", true, mb_port::EXTERNAL);  // Transmit
  d_rx = define_port("rx0", "tmac-rx", true, mb_port::EXTERNAL);  // Receive
  d_cs = define_port("cs", "tmac-cs", true, mb_port::EXTERNAL);   // Control/Status
}

// Invoked when the base 'mac' class finishes initializing the USRP
void tmac::usrp_initialized()
{
  initialize_tmac();
 
  // Send any information between the MAC and the PHY, require max frame sizes
  d_mac_properties = pmt_make_dict();
  pmt_dict_set(d_mac_properties, pmt_intern("max-frame"), pmt_from_long(MAX_FRAME_SIZE));
  pmt_dict_set(d_mac_properties, pmt_intern("max-payload"), pmt_from_long(max_frame_payload()));
}

void tmac::initialize_tmac()
{
  // The base station takes special initialization
  if(d_base_station) {
    initialize_base_station();
  
    d_cs->send(s_response_mac_initialized,                  // Notify the application that
               pmt_list3(PMT_NIL, PMT_T, d_mac_properties));  // the MAC is initialized
  } else {
    // Regular node
    initialize_node();
    if(verbose)
      std::cout << "[TMAC] Waiting for SYNC before notifying application of initialization...\n";
  }
}

// This is the crux of the MAC layer.  The general architecture is to have
// states at which the MAC is in, and then handle messages from the application
// and physical layer based on that state.  
// 
// Then, the incoming 'event' (type of message) in the state triggers some
// functionality.  For example, if we are in the idle state and receive a 
// s_cmd_tx_data event from the application (detected by the port ID), we send
// the data to the physical layer to be modulated.
void tmac::handle_mac_message(mb_message_sptr msg)
{
  pmt_t event = msg->signal();      // type of message
  pmt_t data = msg->data();         // the associated data
  pmt_t port_id = msg->port_id();   // the port the msg was received on

  std::string error_msg;

  switch(d_state) {
    
    //----------------------------- INIT TMAC --------------------------------//
    // In the INIT_TMAC state, now that the USRP is initialized.
    case INIT_TMAC:
      goto unhandled;

    //----------------------------- WAIT SYNC -------------------------------//
    // In this state, we're waiting for a synchronization frame from the
    // basestation so that we can aligned our round and slot.
    case WAIT_SYNC:

      //---- Port: GMSK CS -------------- State: IDLE -----------------------//
      if(pmt_eq(d_phy_cs->port_symbol(), port_id)) {
        
        if(pmt_eq(event, s_response_demod)) {
          incoming_data(data);                        // Incoming data
        }
        return;
      }
      goto unhandled;
    
    //----------------------------- IDLE ------------------------------------//
    // In the idle state the MAC is not quite 'idle', it is just not doing
    // anything specific.  It is still being passive with data between the
    // application and the lower layer.
    case IDLE:
      
      //---- Port: TMAC TX -------------- State: IDLE -----------------------//
      if(pmt_eq(d_tx->port_symbol(), port_id)) {

        if(pmt_eq(event, s_cmd_tx_data)) {
          build_frame(data);
        }
        return;
      }
      
      //---- Port: TMAC CS -------------- State: IDLE -----------------------//
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
          incoming_data(data);                        // Incoming data
        }
        return;
      }
      goto unhandled;

  } // End of switch()
  
 // Received an unhandled message for a specific state
 unhandled:
  if(0 && verbose && !pmt_eq(event, pmt_intern("%shutdown")))
    std::cout << "[TMAC] unhandled msg: " << msg
              << "in state "<< d_state << std::endl;
}

// This method is used for initializing the MAC as a base station, where it will
// transmit synchronization frames at the start of each round.
void tmac::initialize_base_station()
{
  // The base station automatically enters IDLE.  All other nodes wait for SYNC.
  d_state = IDLE;

  // The base station does not decode any frames, so we disable the RX
  disable_rx();

  // The base station specifies the guard time, which it will transmit in its
  // synchronization packets.  Other nodes will calculate parameters on sync.
  calculate_parameters();
 
  // Initialize the next time its going to transmit, which is the initial
  // synchronization frame time
  d_next_tx_time = INITIAL_SYNC;

  // Schedule the initial synchronization frame.
  transmit_sync();
  transmit_sync();
  transmit_sync();
  transmit_sync();
}

// Here we calculate the crucial TDMA parameters.  Everything is calculated in
// clock cycles, which is 1/64e6 based on the FPGA clock.
void tmac::calculate_parameters()
{
  // The number of clock cycles a bit transmission/reception takes
  d_clock_ticks_per_bit = (d_usrp_decim * gmsk::samples_per_symbol()) / BITS_PER_SYMBOL;

  // The slot time is fixed to the maximum frame time over the air.
  d_slot_time = (PREAMBLE_LEN + FRAMING_BITS_LEN + (MAX_FRAME_SIZE * BITS_PER_BYTE) + POSTAMBLE_LEN) * d_clock_ticks_per_bit;

  // The local slot offset depends on the local address and slot/guard times.
  // The local address defines the node's slot assignment.  Slot 0 is for the
  // base station.
  d_local_slot_offset = d_local_address * (d_slot_time + d_guard_time);

  // The total round time takes in to account all of the nodes and slot timing.
  // We add one for the base station.
  d_round_time = (d_total_nodes+1) * (d_slot_time + d_guard_time);

  if(verbose)
    std::cout << "[TMAC] Parameters:"
              << "\n   d_clock_ticks_per_bit: " << d_clock_ticks_per_bit
              << "\n   d_slot_time: " << d_slot_time
              << "\n   d_local_slot_offset: " << d_local_slot_offset
              << "\n   d_round_time: " << d_round_time
              << std::endl;
}


// The initialization of a regular station, which waits for a sync before
// anything can really occur.
void tmac::initialize_node()
{
  enable_rx();

  d_state = WAIT_SYNC;
}

// Invoked when we get a response that a packet was written to the USRP USB bus,
// we assume that it has been transmitted (or will be, within negligable time).
//
// We can notify the application that the transmission was successful if we are
// not the base station.
void tmac::packet_transmitted(pmt_t data)
{
  pmt_t invocation_handle = pmt_nth(0, data);
  pmt_t status = pmt_nth(1, data);
  
  if(!d_base_station)
    d_tx->send(s_response_tx_data,
               pmt_list2(invocation_handle,
                         PMT_T));

  if(d_base_station)
    transmit_sync();

  d_state = IDLE;
}

// Entrance of new incoming data
void tmac::incoming_data(pmt_t data)
{
  pmt_t bits = pmt_nth(0, data);
  pmt_t demod_properties = pmt_nth(1, data);
  std::vector<unsigned char> bit_data = boost::any_cast<std::vector<unsigned char> >(pmt_any_ref(bits));

  // The basestation does not care to decode anything
  if(d_local_address != 0)
    framer(bit_data, demod_properties);
}

// We transmit a sync every round time with an offset of 0, meaning the base
// station will be the start of each round.
void tmac::transmit_sync()
{
  size_t ignore;
  char data;
  
  // Make the PMT data, get a writable pointer to it, then copy our data in
  pmt_t uvec = pmt_make_u8vector(max_frame_payload(), 0);
  d_sync_frame_data *sframe = (d_sync_frame_data *) pmt_u8vector_writable_elements(uvec, ignore);
  char *pay = (char *) pmt_u8vector_writable_elements(uvec, ignore);
  
  for(int i=sizeof(d_sync_frame_data); i<max_frame_payload(); i++)
    pay[i] = 'a';

  // Set the SYNC frame properties
  sframe->guard_time = d_guard_time;
  sframe->total_nodes = d_total_nodes;
  
  // Set the timestamp to the next tx time, and then recalculate the next TX
  // time to be the current time plus a total round time.
  pmt_t timestamp = pmt_from_long(d_next_tx_time);
  d_next_tx_time += d_round_time;

  // Per packet properties
  pmt_t tx_properties = pmt_make_dict();
  pmt_dict_set(tx_properties, pmt_intern("timestamp"), timestamp);
  pmt_t pdata = pmt_list4(PMT_NIL,                        // Unused invoc handle.
                          pmt_from_long(0xffffffff),      // To broadcast.
                          uvec,                           // With data.
                          tx_properties);                 // Properties

  std::cout << "[TMAC] Transmitting SYNC at " << (unsigned long)pmt_to_long(timestamp) << std::endl;
  fflush(stdout);

  build_frame(pdata);
}

// An incoming frame from the physical layer for us!  We check the packet
// properties to determine the sender and if it passed a CRC check, for example.
void tmac::incoming_frame(pmt_t data)
{
  pmt_t invocation_handle = PMT_NIL;
  pmt_t payload = pmt_nth(0, data);
  pmt_t pkt_properties = pmt_nth(1, data);

  // Let's do some checking on the demoded frame
  long src=0, dst=0;

  // Properties are set in the physical layer framing code
  src = pmt_to_long(pmt_dict_ref(pkt_properties,
                                  pmt_intern("src"),
                                  PMT_NIL));

  dst = pmt_to_long(pmt_dict_ref(pkt_properties,
                                  pmt_intern("dst"),
                                  PMT_NIL));

  // All frames from node 0 are SYNC frames, special handling
  if(src == 0) {
    incoming_sync(data);
    return;
  }

  if(dst != d_local_address)  // not for this address
    return;
  
  d_rx->send(s_response_rx_pkt, pmt_list3(invocation_handle, payload, pkt_properties));

  if(verbose)
    std::cout << "[TMAC] Passing up demoded frame\n";
}

// Read an incoming SYNC frame
void tmac::incoming_sync(pmt_t data)
{
  pmt_t invocation_handle = PMT_NIL;
  pmt_t payload = pmt_nth(0, data);
  pmt_t pkt_properties = pmt_nth(1, data);

  unsigned long timestamp = pmt_to_long(pmt_dict_ref(pkt_properties, pmt_intern("timestamp"), PMT_NIL));

  // Cast the incoming payload into SYNC frame data
  size_t ignore;
  d_sync_frame_data *sframe = (d_sync_frame_data *) pmt_u8vector_writable_elements(payload, ignore);

  // Calculate the parameters at the local node
  d_guard_time = sframe->guard_time;
  d_total_nodes =  sframe->total_nodes;
  
  if(verbose && d_state==WAIT_SYNC)
    std::cout << "[TMAC] Received SYNC:"
              << "\n   Timestamp: " << timestamp
              << "\n   Guard Time: " << sframe->guard_time
              << "\n   Total Nodes: " << sframe->total_nodes
              << std::endl;

  // Calculate the local node's first TX time, skip ahead
  if(d_state == WAIT_SYNC) {
    calculate_parameters();
    d_next_tx_time = timestamp + d_local_slot_offset + 100*d_round_time;
    std::cout << "next: " << d_next_tx_time << std::endl;
    
    pmt_dict_set(d_mac_properties, pmt_intern("round-time"), pmt_from_long(d_round_time));
    d_cs->send(s_response_mac_initialized,                  // Notify the application that
               pmt_list3(PMT_NIL, PMT_T, d_mac_properties));  // the MAC is initialized

    d_state = IDLE;
  }

}

REGISTER_MBLOCK_CLASS(tmac);
