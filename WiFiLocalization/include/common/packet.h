/** packet.h: Packet class
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

#ifndef NRL_PACKET_H
#define NRL_PACKET_H

#include <arpa/inet.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <iostream>
#include <vector>

#include "templates.h"
#include "timestamp.h"
#include "location.h"
#include "macaddr.h"
#include "switch.h"

using namespace std;

#define USING_AUXTIME

///Flags for determining packet type
#define NRL_PKT_NOFLAGS		0x00
#define NRL_PKT_CALIBRATE	0x80
#define NRL_PKT_HELLO		0x40
#define NRL_PKT_NOAUXTIME	0x20


class Packet : public Printable, public Serializable {
#if NRL_SWITCH_SOLVER
  friend bool pktcompare(Packet*, Packet*);
  friend class Measurement;
#endif
#if NRL_SWITCH_SIMULATOR
  friend class PacketGen;
  friend class Measurement;
#endif
private:
  MacAddr* mac;
  Timestamp* toa;
  float rssi; //outbound recieved signal strength indication
  Location* sensorloc;
  Timestamp* auxtime; //For test code that sends a time of transmission
  
  double calibdist; //NOTE: Only used by solver. NOT transmitted via networks.
public:
  char flags; //For indicating packet type, such as "calibration packet"
  
  void pktbuilder(MacAddr*, Timestamp*, float, Location*, char flag);
  Timestamp* timediff(void); //Difference between ToA and auxtime
  
  void print(ostream* fs);
  void printcsv(ostream* fs);
  Packet* copy(void);
  //Packet constructors. If auxtime is in use, constructors that don't use it will set auxtime to NULL.
  //If auxtime is not in use, constructors that do use it will ignore the argument (as though it were NULL).
  Packet(MacAddr*, Timestamp*, float, Location*);
  Packet(MacAddr*, Timestamp*, float, Location*, char);
  Packet(MacAddr*, Timestamp*, float, Location*, Timestamp*);
  Packet(MacAddr*, Timestamp*, float, Location*, Timestamp*, char);
  
  Location* getloc(void) { return sensorloc->copy(); };
  Location getloc2(void) { return *sensorloc; }; //BUG doesn't check to see if valid ptr
  float getrssi(void) { return rssi; };
  double getcalibdist(void) { return calibdist; };
  
  //Serialize packet for network transmission
  int serialize(char* buf, int buflen);
  Packet(char* buf, int buflen); //constructor for unserialization
  //for packet aggregation
  bool isMatch(Timestamp* time_toler, Packet* other);
  //std::vector<Packet*>* findMatches(Timestamp* time_toler, std::vector<Packet*> *pktlist,
//					std::vector<bool> *matched, int my_iter);
  bool isATestPkt(void);
  bool isCalibrationPkt(void);
  bool isExpiring(Timestamp* time_to_live, Timestamp* last_collection_time); //Check to see if it's expiring (aggregation algorithm)
  ///setCalibration: Using list of MAC addresses specified for calibration,
  ///	determine if this is a calibration packet
  ///Pre: macvec contains valid MAC addresses or NULL (0 is allowed)
  ///	locvec contains corresponding locations
  ///Post: if this->mac matches an address in macvec, set NRL_FLAG_CALIBRATE
  ///	and save the actual distance for this packet in 'calibdist'
  ///Complexity: O( size of macvec )
  void setCalibration(vector<MacAddr*> macvec, vector<Location*> locvec); 
};

#endif
