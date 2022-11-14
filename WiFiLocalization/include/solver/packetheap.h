/** packetheap.h: A heap of packets arranged from oldest to newest timestamp
 * used by the aggregator to keep track of what it has
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
#ifndef PKT_HEAP_H
#define PKT_HEAP_H

#include "packet.h"
#include "measurement.h"
#include <algorithm>
#include <vector>
#include <deque>

using namespace std;

/// PacketHeap: A min-heap of packets
/// used by the aggregator to keep track of packets it recieves
class PacketHeap{
private:
  std::vector<Packet*> pktheap;
  void makeMeasurementHelper(std::vector<Packet*>* matchedpkts, std::vector<Measurement*>* measlist); //Helper function for groupPkts()
public:
  PacketHeap(void);
  ~PacketHeap(void);
  void append(Packet* pkt);
  void append(std::vector<Packet*>* pktlist);
  void append(std::deque<void*>* pktlist);
  std::vector<Measurement*>* groupPkts(Timestamp* time_toler,
				       Timestamp* time_to_wait,
				       int num_sensors); //AGGREGATION ALGORITHM!
};

#endif
