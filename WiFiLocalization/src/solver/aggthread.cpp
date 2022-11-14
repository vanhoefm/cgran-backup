/** aggregator.cpp: Packet aggregator thread with server
 * Retrieves packets from the sockets, aggregates them when not recieving
 * Produces an event when a packet group is ready
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

//Note: This thread is designed to be a joinable thread allocated on the heap.

#include "aggthread.h"
#include <deque>
#include <wx/thread.h>

using namespace std;


/*---------AggParams implementation-------*/

void AggParams::print(ostream* fs){
  *fs << "/-----Aggregator parameters-----\\\n";
  *fs << "Time tolerance between matching timestamps: ";
  time_toler->print(fs);
  *fs << "\nTime for packets to live in aggregator buffer: \n";
  time_to_wait->print(fs); //Actually, time for network transmission and buffer combined
  *fs << "Number of sensors: " << num_sensors << endl;
  *fs << "\\-------------------------------/\n";
}

void AggParams::printcsv(ostream* fs){
  time_toler->printcsv(fs);
  *fs << ";";
  time_to_wait->printcsv(fs);
  *fs << ";" << num_sensors;
}

AggParams* AggParams::copy(void){
  AggParams* mycopy = new AggParams();
  //Make duplicate copies of all data
  mycopy->time_toler = new Timestamp(time_toler);
  mycopy->time_to_wait = new Timestamp(time_to_wait);
  mycopy->num_sensors = num_sensors;
  return mycopy;
}


/*---------AggThread implementation---------*/

/// Thread constructor used to pass in and store arguments 
// (only one is needed)
AggThread::AggThread(CondBuffer* buf, CondBuffer* solverbuf, ParamsWrapper* apwrap) : wxThread(wxTHREAD_JOINABLE){
  buffer = buf;
  solverbuffer = solverbuf;
  pwrap = apwrap;
  done = false;
}


/// Thread execution function
void* AggThread::Entry(void){
  
  cout << "INFO: aggregator thread running" << endl;
  
  int a;
  storageheap = new PacketHeap();
  std::deque<void*>* items;
  std::vector<Measurement*>* measurements;
  
  while (!done){
    //Grab everything in the buffer
    //cout << "About to grab from buffer (aggregator thread)" << endl;
    items = buffer->remove(AGGREGATOR_QUEUE_DELAY, This(), true); //Blocks when buffer is empty
    
    try {
      //Add recovered packets (if any) to the heap.
      if (items->size() > 0)
	storageheap->append(items);
      
      //Find expiring packets and turn them into measurements
      //cout << "Aggregator about to get params" << endl;
      Params* params = pwrap->getParams();
      AggParams* params2;
      if (params->getType() == PT_AGG) //Typecheck
	params2 = (AggParams*) params;
      else
	throw "Aggregator: aggregator parameters is not the correct type of Params!";
      
      //cout << "Aggregator params struct obtained" << endl;
      
      Timestamp* time_toler = params2->time_toler; //Matching time tolerance
      Timestamp* time_to_wait = params2->time_to_wait; //Time between when sensor collects a timestamp and when the aggregator says it's expired.
      int num_sensors = params2->num_sensors;
      
      //cout << "Aggregator about to do its job" << endl;
      //Aggregate!
      measurements = storageheap->groupPkts(time_toler, time_to_wait, num_sensors);
      
      if (measurements->size() > 0){
	//Add measurements to solver's buffer
	//cout << "About to add measurements to solver's buffer" << endl;
	vector<void*> voidvec;
	voidvec.reserve(measurements->size());
	int a;
	for (a = 0; a < measurements->size(); a++)
	  voidvec.push_back( (void*) (*measurements)[a] );
	
	solverbuffer->add(voidvec);   
	//cout << "Successfully added measurements to solver's buffer" << endl;
	
      }
      //cout << "About to delete measurements" << endl;
      delete measurements;
      
    } catch (const char* err){
      cout << "Aggregator thread error: " << err << endl;
    } catch (...){
      cout << "Other error in aggregator thread" << endl;
    }
    
    done = done || TestDestroy(); //REQUIRED: Check to see if the thread needs to exit   
  }
  
  //Exit and join with main thread
  delete storageheap;
  delete pwrap;
  return NULL;
}


void AggThread::signalDone(void){
  done = true;
}


