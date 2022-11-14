/** packetlist.cpp: A wrapper for packet vectors
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
//NOTE: It is assumed only the aggregator thread uses this class.

//FIXME: Move this definition
#define MIN_NUM_MEASUREMENTS 	3

#include "packetheap.h"

#include <functional>

//Packet comparison function, compliant with <algorithm>'s "make_heap" function.
//Used to determine which packet should go first in the heap (a.k.a. which is older)
bool pktcompare(Packet* a, Packet* b){
  Timestamp atime = *(a->toa);
  Timestamp btime = *(b->toa);
  return atime < btime;
}

struct Compare : public binary_function<Packet*,Packet*,bool> {
  bool operator() (Packet* a, Packet* b) {return (pktcompare(a, b));}
};

///@arg aggparams: the initial aggregator parameters
PacketHeap::PacketHeap(void){
  //An empty vector is already a heap, so we don't need to make_heap pktheap.
}

PacketHeap::~PacketHeap(void){
  int a;
  if (pktheap.size() > 0)
    cout << "Warning! " << pktheap.size() << " packets in aggregator's heap when flushed." << endl;
  
  for (a = 0; a < pktheap.size(); a++)
    delete pktheap[a];   
}


//Add item to the heap (O(lg N) where N is the number of items in the heap already)
void PacketHeap::append(Packet* pkt){
  pktheap.push_back(pkt);
  push_heap(pktheap.begin(), pktheap.end(), Compare());
}

//Add items to heap (O(M lg N) where M is number of items in pktlist, N is number of items in heap)
void PacketHeap::append(std::vector<Packet*>* pktlist){
  int a;
  for (a = 0; a < pktlist->size(); a++)
    append((*pktlist)[a]);
}

//Add items to heap (O(M lg N) where M is number of items in pktlist, N is number of items in heap)
void PacketHeap::append(std::deque<void*>* pktlist){
  int a;
  for (a = 0; a < pktlist->size(); a++)
    append((Packet*) (*pktlist)[a]);
}


/// Perform packet aggregation
std::vector<Measurement*>* PacketHeap::groupPkts(Timestamp* time_toler,
						 Timestamp* time_to_wait,
						 int num_sensors){
  
  std::vector<Measurement*> *measlist = new std::vector<Measurement*>();
  Timestamp now; //Current time.
  
  if (num_sensors < MIN_NUM_MEASUREMENTS)
    throw "Cannot groupPkts: not enough sensors (need 3 or more)!";

  std::vector<Packet*> expiringpkts;
  unsigned int a;
  
  //Look for all packets due to expire. O(n lg n) where n is the number of items (worst case: they're all expiring) 
  bool isexpiring = true;
  while (isexpiring && (pktheap.size() > 0)){
    //Check to see if first item is expiring. This is a heap, so all expiring items appear first.
    isexpiring = pktheap[0]->isExpiring(time_to_wait, &now);
    if (isexpiring){
      pktheap[0]->print(&cout);
      expiringpkts.push_back(pktheap[0]);
      pop_heap(pktheap.begin(), pktheap.end(), Compare());
      pktheap.pop_back();
    }
  }
  
  std::vector<Packet*> matchedpkts;
  matchedpkts.reserve(num_sensors);
  //Now that we have all expiring packets, find matches. 
  //Invariant: The vector of expiring packets is in sorted order: oldest packet first.
  //Start at the end of the array and work towards the beginning, grouping packets.
  //O(M * N) where M is the number of sensors and N is the number of expiring packets.
  //This should be able to handle lost packets, but not duplicate packets (duplicates should never occur)
  if (expiringpkts.size() >= num_sensors){
    //cout << expiringpkts.size() << " packets are expiring!" << endl;
    int b = 0; //index of the packet being matched with others in the list.
    matchedpkts.push_back(expiringpkts[b]); //Save the packet we are matching with others.
    for (a = 1; a < expiringpkts.size(); a++){
      bool ismatch = expiringpkts[b]->isMatch(time_toler, expiringpkts[a]);
      if (ismatch){
	//Save match and continue looking for more
	matchedpkts.push_back(expiringpkts[a]);
      } else {
	//Next previous packet is not a match. Save current list of matches as a measurement.
	//Make the new measurement
	makeMeasurementHelper(&matchedpkts, measlist);
	matchedpkts.clear();
	b = a;//Use the non-match as the comparison packet for the next group of matches.
	matchedpkts.push_back(expiringpkts[b]); //Save the packet we are matching with others
      }
    }
    makeMeasurementHelper(&matchedpkts, measlist);
    
    //cout << "Done matching packets with one another." << endl;
    
  } else {
    if (expiringpkts.size() != 0)
      cout << "Aggregator dropping " << expiringpkts.size() << " packets that can't be matched." << endl;
  }
  
  //cout << "Deleting expired packets. Measurement keeps its own copies" << endl;
  for (a = 0; a < expiringpkts.size(); a++)
    delete expiringpkts[a];
  
  return measlist;
}

//Helper function for groupPkts()
void PacketHeap::makeMeasurementHelper(std::vector<Packet*>* matchedpkts, std::vector<Measurement*>* measlist){
  Measurement* meas = new Measurement(); //NOTE: This MUST make its own copies of the packets.
  bool valid = meas->measbuilder(matchedpkts);
  if (valid) {
    //cout << "Adding measurement to list" << endl;
    measlist->push_back(meas);
  } else {
    cout << "Measurement invalid" << endl;
    delete meas;
  }
}

