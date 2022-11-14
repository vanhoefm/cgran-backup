/** pktgen.cpp: Packet generator for the simulator
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
#include <cmath>

#include "pktgen.h"
#include "pktsend.h"

using namespace std;


//Packet* PacketGen::genPacket(Location* src, Location* dest, Timestamp* time){
Packet* PacketGen::genPacket(Intruder* src, Sensor* dest, Timestamp* time){
  
  //FIXME: No check to see if the sensor is within range of the transmitter
  
  /*
  /// Check: Did this packet get dropped?
  double lost_pkt = generate_variance(1);
  if (lost_pkt < gparams.p_lost_pkt) //FIXME: gparams is no longer in use
    return NULL; //It got dropped
  */
  
  /// Calculate correct values for all Packet fields
  double dist = src->lcn.distanceToDest(dest->loc);
  cout << "Distance generated (without noise) is " << dist << endl;
  Timestamp* toa = toamodel->dist_to_toa(dist, time);
  
  //new Timestamp(time->gettime(), time->getnsec() + dist / ((double) C_NSEC));
  float rssi = rssimodel->dist_to_rssi(dist);
  
  /// Add noise and other errors to data
 
  
  /// Build packet
  char flags = NRL_PKT_NOFLAGS;
  Packet* pkt;
#ifdef USING_AUXTIME //defined (or not) in Packet.h
  pkt = new Packet(src->mac, toa, rssi, dest->loc, time, flags);
#else
  pkt = new Packet(src->mac, toa, rssi, dest->loc, flags);
#endif
  
  //Cleanup
  delete toa;
  
  //cout << "Generated packet:" << endl;
  //pkt->print(&cout);
  
  return pkt;
}


PacketGen::PacketGen(PathLossModel* rmodel, ToADistanceCalculator* tmodel){
  rssimodel = rmodel;
  toamodel = tmodel;
}

void PacketGen::newModel(PathLossModel* rmodel, ToADistanceCalculator* tmodel){
  PathLossModel* temp = rssimodel;
  rssimodel = rmodel; //Must be atomic if this function is to be considered thread safe
  delete temp;
  ToADistanceCalculator* temp2 = toamodel;
  toamodel = tmodel; //Must be atomic if this function is to be considered thread safe
  delete temp2;
}

PacketGen::~PacketGen(void){
  delete rssimodel;
  delete toamodel;
}

