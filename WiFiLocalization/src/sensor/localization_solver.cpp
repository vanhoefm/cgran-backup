/** localization_solver.cpp
 * 
 * An interface between the BBN 802.11 PLCP module and the localization solver
 * 
 * This could later be turned into an mblock sink
 * 
 * @author Brian Shaw
 * 
 */
/* 
 * This file is part of WiFi Localization
 * 
 * This program is free software; you can redistribute it and/or modify 
 * it under the terms of the GNU General Public License as published by 
 * the Free Software Foundation; either version 2 of the License, or 
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but 
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License 
 * for more details.
 * 
 * You should have received a copy of the GNU General Public License along 
 * with this program; if not, write to the Free Software Foundation, Inc., 
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

#include <iostream>

#include "localization_solver.h"
#include "packet.h"
#include "sensorsocket.h"
#include "sensor_globals.h"
#include "sensor_threads.h"
#include "read_params.h"

#include <stdlib.h>

//Ryan's packet format and USING_AUXTIME both need to be enabled (or not)
//Make it easy, have both get enabled in packet.h by enabling USING_AUXTIME
#ifdef USING_AUXTIME
  #define NRL_USING_RYAN80211
#else
  #define NRL_USING_IEEE_80211
#endif

using namespace std;

char senderhost[256];
int senderport;
Location myloc;

//Globals for use in this file only (not externed into BBN code)
TCPClientConnection* tcplink;
//Globals storing pointers to packet buffers
Pktbuffer* senderbuf;
Pktbuffer* loggerbuf;

//Thread ID globals
pthread_t senderthread;
pthread_t loggerthread;

///Main, BBN-called packet passing function
//void packet_to_solver(unsigned char* d_pkt_data, int d_pdu_len, long long d_packet_rx_time, float rssi){
void NrlSolver::passpkt(unsigned char* d_pkt_data, int d_pdu_len, long long d_packet_rx_time, float rssi){
  //The same names are used as within the PLCP file for clarity
  //d_pkt_data is a pointer to a 2500-long character array that's about to get free()'d
  
  /* This method doesn't make accurate timestamps
  //cout << "Packet received at " << d_packet_rx_time << endl;
  //Get timestamp info (This retrieves millisecond-accurate timestamps)
  //BBN80211 timestamps are in microseconds since the receiver started
  int64_t sec = (int64_t) d_packet_rx_time / 1000000;
  double nsec = (double) (d_packet_rx_time - sec * 1000000) * 1000;
  Timestamp startts(starttime.tv_sec, (double) starttime.tv_usec * 1000);
  Timestamp timearrived(sec, nsec);
  Timestamp* toa = startts + timearrived;
  */
  
  //Collect timestamp for this packet
  Timestamp toa; //Automatically obtains ns-precise timestamp
  
#ifdef NRL_USING_IEEE_80211
  //cout << "Called packet_to_solver" << endl;
  //NOTE: We assume the real mac address format is in use
  //NOTE: We are using Address 1, the source address, assuming it is not spoofed.
  MacAddr* mac;
  char buf[7];
  bool worked = false;
  if (d_pdu_len >= 10){
    memcpy(buf, (char*) &d_pkt_data[4], 6 * sizeof(char));
    //buf[6] = '\0'; //Null terminate so strlen() doesn't fritz out
    mac = new MacAddr();
    worked = mac->macBuilder(buf, 6);
  } else {
    mac = NULL;
    worked = false;
  }
  
  //Check for mac errors
  if (!worked){
    cout << "Mac Address 1 read in incorrectly. Dropping packet." << endl;
    return;
  } else {
    cout << "Source ";
    mac->print(&cout); //Debug
  }
    
  //BEGIN DEBUG
  /*
  MacAddr* mac2;
  MacAddr* mac3;
  MacAddr* mac4;
  
  if (d_pdu_len >= 16){
    memcpy(buf, (char*) &d_pkt_data[10], 6 * sizeof(char));
    buf[6] = '\0'; //Null terminate so strlen() doesn't fritz out
    mac2 = new MacAddr();
    worked = mac2->macBuilder(buf, 6);
  } else {
    mac2 = NULL;
    worked = false;
  }
  
  //Check for mac errors
  if (!worked){
    cout << "Mac Address 2 read in incorrectly. Dropping packet." << endl;
    return;
  } else {
    cout << "(final) Destination ";
    mac2->print(&cout); //Debug
  }
  
  if (d_pdu_len >= 22){
    memcpy(buf, (char*) &d_pkt_data[16], 6 * sizeof(char));
    buf[6] = '\0'; //Null terminate so strlen() doesn't fritz out
    mac3 = new MacAddr();
    worked = mac3->macBuilder(buf, 6);
  } else {
    mac3 = NULL;
    worked = false;
  }
  
  //Check for mac errors
  if (!worked){
    cout << "Mac Address 3 read in incorrectly. Dropping packet." << endl;
    return;
  } else {
    cout << "Receiver ";
    mac3->print(&cout); //Debug
    worked = false;
  }
  
  if (d_pdu_len >= 30){
    memcpy(buf, (char*) &d_pkt_data[24], 6 * sizeof(char));
    buf[6] = '\0'; //Null terminate so strlen() doesn't fritz out
    mac4 = new MacAddr();
    worked = mac4->macBuilder(buf, 6);
  } else {
    mac4 = NULL;
  }
  
  //Check for mac errors
  if (!worked){
    cout << "Mac Address 4 read in incorrectly. Dropping packet." << endl;
    return;
  } else {
    cout << "Transmitter ";
    mac4->print(&cout); //Debug
  }
  */
  //END DEBUG
  
  cout << "used IEEE format" << endl;
  
  Packet* pkt = new Packet(mac, &toa, rssi, &myloc); //doesn't use auxtime
#endif
#ifdef NRL_USING_RYAN80211
  
  MacAddr* mac;
  char buf[22];
  bool worked = false;
  if (d_pdu_len >= 37){
    memcpy(buf, (char*) &d_pkt_data[0], 17 * sizeof(char));
    //buf[17] = '\0'; //Null terminate so strlen() doesn't fritz out--FIXME is this even needed? If so, put it in the macBuilder() function.
    mac = new MacAddr();
    worked = mac->macBuilder(buf, 17);
  } else {
    mac = NULL;
    worked = false;
    cout << "Packet not long enough." << endl;
  }
  
  //Check for mac errors
  if (!worked){
    cout << "Packet read in incorrectly. Dropping packet." << endl;
    return;
  } else {
    cout << "Source ";
    mac->print(&cout); //Debug
  }
  
  strncpy(buf, (char*) &d_pkt_data[17], 20);
  buf[20] = '\0';
  cout << "Time of Tx: " << buf << endl;
  Timestamp txtime;
  txtime.timeBuilder(buf, 22);
  
  
  //NOTE: This read-in segfaults when data is corrupt.
  //Timestamp* txtime = new Timestamp((char*) &d_pkt_data[17]);
  
  txtime.print(&cout);
  cout << "RSSI: " << rssi << endl;
  
  Packet* pkt = new Packet(mac, &toa, rssi, &myloc, &txtime);
  //delete txtime;
  
#endif
  
  
  int err;
  
#if TCP_SENDING_ENABLED
  Packet* pkt2 = pkt->copy();
  err = senderbuf->add_packet(pkt2, true);
  if (err)
    cout << "Warning: Problem adding packet to sender thread's buffer" << endl;
#endif
  
  err = loggerbuf->add_packet(pkt, true);
  if (err)
    cout << "Warning: Problem adding packet to logger thread's buffer" << endl;
  
  delete mac;
  /*
  delete mac2;
  delete mac3;
  delete mac4;
  */
  //delete toa;
  
  //cout << "Done with giving a packet to the sender" << endl;
}


///BBN-called initialization function
//void localization_solver_startup(void){
NrlSolver::NrlSolver(){
  
  myloc = nrl_read_params(senderhost, 256, &senderport);
  
  //Setup TCP stuff. If it fails, why bother creating the threads?
  #if TCP_SENDING_ENABLED
  tcplink = new TCPClientConnection(senderhost, senderport, false, false);
  //tcplink = new TCPClientConnection(senderhost, senderport, IS_SOLVER_LOCAL, false);
  #endif
  
  //Setup packet buffer and thread
  //cout << "Making packetbuffers" << endl;
  senderbuf = new Pktbuffer(BBNSENDER_PKTBUF_SIZE);
  loggerbuf = new Pktbuffer(BBNSENDER_PKTBUF_SIZE);
  
  //Setup helper threads: logger and sender
  //Note: Both structs are the same and only need the time_to_run
  struct solverparams* senderparams = new struct solverparams;
  struct solverparams* loggerparams = new struct solverparams;
  senderparams->packetbuf = senderbuf;
  loggerparams->packetbuf = loggerbuf;
#if TCP_SENDING_ENABLED
  senderparams->link = tcplink;
#endif
  
  istimetodie = new Atomic(0); //When this gets set to 1, the threads will stop running. I hope.
  
  //FIXME: Signals should be used to kill the threads when done, not timeouts.
  //Note: Other parameters exist in struct solverparams, but are unused by these threads.
  //cout << "Making sender threads" << endl;
  int thread_error = pthread_create(&loggerthread, NULL, logger_thread, (void*) loggerparams);
  if (thread_error){
    cout << "Error: Cannot create packet logger thread" << endl;
    exit(1);
  }
#if TCP_SENDING_ENABLED
  thread_error = pthread_create(&senderthread, NULL, pktsender_thread, (void*) senderparams);
  if (thread_error){
    cout << "Error: Cannot create packet sending thread" << endl;
    exit(1);
  }
#endif
  
  cout << "Sender fully initialized" << endl;
  //return;
}


NrlSolver::~NrlSolver(){
//void localization_solver_shutdown(void){
  cout << "Closing sender" << endl;
  istimetodie->setval(1); //Set the flag for "you should stop running now"
  
  pthread_join(loggerthread, NULL);
  pthread_join(senderthread, NULL);
  //return;
}


void NrlSolver::setstarttime(void){
  int err = gettimeofday(&starttime, NULL);
  if (err)
    cout << "Warning! Unable to set start time." << endl;
}


