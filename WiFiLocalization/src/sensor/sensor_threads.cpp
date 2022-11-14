/** sensor_threads.cpp: Sender and Logger thread functionality for sensor
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

#include "sensor_globals.h"
#include "sensor_threads.h"
#include "sensorsocket.h"
#include "pktbuffer.h"

#include <iostream>
#include <fstream>
#include <unistd.h>
#include <stdlib.h>

using namespace std;

void* logger_thread(void* arg);
void init_threads(int buffer_capacity, bool log_environment_pkts);
void stop_threads(void);

Atomic* istimetodie;

#if TCP_SENDING_ENABLED
//Helper thread for handling I/O
//Threading is used to ensure that blocking I/O will not destroy reciever performance.
//Note: TCP client socket has already been set up during initialization
//NOTE: This code based heavily on the logger from packet.cpp
void* pktsender_thread(void* paramstruct){
  struct solverparams* params = (struct solverparams*) paramstruct;
  Pktbuffer* loggerbuf = params->packetbuf;
  TCPClientConnection* tcplink = params->link;
  char buf[256];
  
  cout << "Packet sender thread running" << endl;
  
  //until it's time to be finished, send available packets
  while (! istimetodie->getval()){
    //Once every 1/10th of a second, grab all available packets and write them to disk
    Packet* pkt = loggerbuf->get_packet(true);
    while (pkt != NULL){
      //Send it via TCP
      cout << "Sender got a packet" << endl;
      pkt->serialize(buf, 256);
      send_over_link(buf, 256, tcplink);
      //cout << "About to delete packet" << endl;
      delete pkt;
      pkt = loggerbuf->get_packet(true);
    }
    usleep(UPDATE_RATE * 1000); //wait 100,000 us
  }
  
  return NULL;
}
#endif


/*-----------LOGGER FUNCTIONS-------------*/

//Thread for packet logging
//NOTE: Packet log's name is hard-coded
void* logger_thread(void* arg){
  struct solverparams* params = (struct solverparams*) arg;
  
  cout << "Logger thread running" << endl;
  
  ofstream report;
  report.open("pktlog.txt", ios_base::out | ios_base::trunc);
  if (!report.is_open()){
    cout << "Unable to open log file" << endl;
    exit(1);
  }
  
  ofstream report2;
  report2.open("pktlog.csv", ios_base::out | ios_base::trunc);
  if (!report2.is_open()){
    cout << "Unable to open csv file" << endl;
    exit(1);
  }
  
  //until it's time to be finished... perhaps a while(1) loop with a "quit" signal to exit?
  while (! istimetodie->getval()){
    //Once every 1/10th of a second, grab all available packets and write them to disk
    Packet* pkt = params->packetbuf->get_packet(true);
    while (pkt != NULL){
      //cout << "Logger got a packet!" << endl;
      //Write it out to disk
      //pkt->print(&report);
      pkt->printcsv(&report2);
      report2 << endl;
      delete pkt;
      //cout << "Logged a packet." << endl;
      pkt = params->packetbuf->get_packet(true);
    }
    usleep(UPDATE_RATE * 1000); //wait 100,000 us
  }
  
  report.close();
  report2.close();
  return NULL;
}
