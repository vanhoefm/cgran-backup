/** packet.cpp: Packet class implementation
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

#include "packet.h"
#include "bitutils.h"

using namespace std;

/*--------------Implementation of Packet class----------------------*/

//Helper function
void Packet::pktbuilder(MacAddr* macaddr, Timestamp* timestamp,
		float rssi_val, Location* spot, char flag){
  
  if (macaddr != NULL)
    mac = new MacAddr(*macaddr);
  else
    throw "Null MAC address recieved by Packet constructor 1\n";
  if (timestamp != NULL)
    toa = new Timestamp(timestamp);
  else
    cerr << "Warning! Null Timestamp recieved by Packet constructor 1\n";
  rssi = rssi_val;
  if (spot != NULL)
    sensorloc = new Location(spot);
  else
    cerr << "Warning! Null Location recieved by Packet constructor 1\n";
  flags = flag;
  calibdist = 0; //Gets set in setCalibration(), if ever. See that function for details
}

/// Packet constructors
  
//Without flags
Packet::Packet (MacAddr* macaddr, Timestamp* timestamp,
		float rssi_val, Location* spot, Timestamp* auxtimme){
  pktbuilder(macaddr, timestamp, rssi_val, spot, NRL_PKT_NOFLAGS);
#ifdef USING_AUXTIME
  if (auxtimme != NULL)
    auxtime = new Timestamp(auxtimme);
  else
    auxtime = NULL;
#else
  auxtime = NULL;
#endif
}

Packet::Packet (MacAddr* macaddr, Timestamp* timestamp, float rssi_val, Location* spot){
  pktbuilder(macaddr, timestamp, rssi_val, spot, NRL_PKT_NOFLAGS);
  auxtime = NULL;
}


//With flags
Packet::Packet (MacAddr* macaddr, Timestamp* timestamp,
		float rssi_val, Location* spot, Timestamp* auxtimme, char flag){
  pktbuilder(macaddr, timestamp, rssi_val, spot, flag);
#ifdef USING_AUXTIME
  if (auxtimme != NULL)
    auxtime = auxtimme->copy();
  else
    auxtime = NULL;
#else
  auxtime = NULL;
#endif
}

Packet::Packet (MacAddr* macaddr, Timestamp* timestamp, float rssi_val, Location* spot, char flag){
  pktbuilder(macaddr, timestamp, rssi_val, spot, flag);
  auxtime = NULL;
}


//Serialize a packet for network transmission
int Packet::serialize(char* buf, int bufsize){
  int timestampsize = getTimestampSize();
  int locationsize = getLocationSize();
  int totalsize = sizeof(int64_t) + timestampsize + sizeof(float) + locationsize + sizeof(char);
  totalsize += timestampsize;
  if (bufsize < totalsize)
    throw "Cannot serialize packet, buffer needs to be larger";
   
  int a = 0;
  //copy in MAC address
  a += mac->serialize(&buf[a], bufsize - a);
  //copy in ToA
  toa->serialize(&buf[a], bufsize - a);
  a += timestampsize;
  //copy in rssi
  nrl_htonf(rssi, &buf[a]);
  a += sizeof(float);
  //copy in location
  sensorloc->serialize(&buf[a], bufsize - a);
  a += locationsize;
  //copy in auxtime
  if (auxtime != NULL)
    auxtime->serialize(&buf[a], bufsize - a);
  else
    flags = flags | NRL_PKT_NOAUXTIME;
  a += timestampsize;
  //copy in flags
  buf[a] = flags;
  a++;
  return a;
}
  

//Generate a packet from serialized network data
//(by overloading the constructor)
//Note: The size of the item being passed in is used to indicate whether or not it has an aux timestamp
Packet::Packet(char* buf, int bufsize){
  int timestampsize = getTimestampSize();
  int locationsize = getLocationSize();
  int totalsize = sizeof(int64_t) + 2 * timestampsize + sizeof(float) + locationsize + sizeof(char);
  if (bufsize < totalsize)
    throw "Cannot unserialize packet, buffer needs to be larger";
  int a = 0;
  //Copy in MAC address
  mac = new MacAddr(&buf[a], bufsize - a);
  a += sizeof(int64_t);
  //copy in ToA
  toa = new Timestamp(&buf[a], bufsize - a);
  a += timestampsize;
  //Copy in RSSI
  rssi = nrl_ntohf(&buf[a]);
  a += sizeof(int);
  //Copying in Location
  sensorloc = new Location(&buf[a], bufsize - a);
  a += locationsize;
  //Copying in Flags (note: copied in from the location after Timestamp)
  flags = buf[a + timestampsize];
  //Copying in AuxTime
  if (flags & NRL_PKT_NOAUXTIME)
    auxtime = NULL;
  else {
    auxtime = new Timestamp(&buf[a], bufsize - a);
    //a += timestampsize;
  }
}


Packet* Packet::copy(void){
  return new Packet(mac, toa, rssi, sensorloc, auxtime, flags);
}


//Time difference (between ToA and auxtime)
Timestamp* Packet::timediff(void){
  if ((toa == NULL) || (auxtime == NULL))
    return NULL;
  else
    return *toa - *auxtime;
}


void Packet::print(ostream* fs){
  //print border
  *fs << "-------------Received Packet-------------\n";
  //print MAC address
  *fs << "Source ";
  mac->print(fs);
  //Print auxtime
  if (auxtime != NULL){
    *fs << "Time sent: ";
    auxtime->print(fs);
  }
  //print toa
  *fs << "Time recieved at sensor: ";
  toa->print(fs);
  //print rssi
  *fs << "RSSI: " << rssi << '\n';
  //print location
  *fs << "Sensor location: ";
  sensorloc->print(fs);
  //Border
  *fs << "-----------------------------------------\n";
}

void Packet::printcsv(ostream* fs){
  mac->printcsv(fs);
  *fs << ";";
  if (auxtime != NULL){
    auxtime->printcsv(fs);
    *fs << ";";
  }
  toa->printcsv(fs);
  *fs << ";" << rssi << ";";
  sensorloc->printcsv(fs);
}


bool Packet::isATestPkt(void){
  return (flags & NRL_PKT_HELLO) == NRL_PKT_HELLO;
}

bool Packet::isCalibrationPkt(void){
  return (flags & NRL_PKT_CALIBRATE) == NRL_PKT_CALIBRATE;
}


/*--------Aggregator helper functions-----------------*/

///Check to see if the other packet is within our time tolerance
bool Packet::isMatch(Timestamp* time_toler, Packet* other){
  bool match = toa->isNear(other->toa, time_toler) && mac->sameMac(other->mac);
  
  //If we have "auxtime" in use with the simulator, both packets' "auxtime" should match
  if (auxtime != NULL && other->auxtime != NULL){
    bool exactsame = auxtime->isSame(other->auxtime);
    //if (match && exactsame)
    //  cout << "Sent time matches, packets came from same source.\n";
    if (match && !exactsame)
      cout << "Warning: Aggregator's match is probably erroneous (different sent times)\n";
  }
  return match;
}


//Check to see if it's expiring (aggregation algorithm)
bool Packet::isExpiring(Timestamp* time_to_live, Timestamp* last_collection_time){
  return toa->isExpiring(time_to_live, last_collection_time);
}


//If mac matches one of the calibration MAC addresses, set calibration flag
void Packet::setCalibration(vector<MacAddr*> macvec, vector<Location*> locvec){
  int a;
  for (a = 0; a < macvec.size(); a++){
    if (mac->sameMac(macvec[a])){
      //MAC address matches one of the calibration MACs!
      cout << "Setting calibration packet" << endl;
      flags = flags | NRL_PKT_CALIBRATE;
      //Store distance from sensor in helper variable (not for network transmission!)
      calibdist = sensorloc->distanceToDest(locvec[a]); //Sets actual distance from intruder to sensor
      return;
    }
  }
  //Not a calibration packet
  return;
}

