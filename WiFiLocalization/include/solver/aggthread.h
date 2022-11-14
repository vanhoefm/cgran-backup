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

#ifndef AGGREGATOR_H
#define AGGREGATOR_H


#include <wx/wx.h>
#include <wx/thread.h>
#include "packetheap.h"
#include "params.h"
#include "buffer.h"

using namespace std;

//Time in ms between updates
#define AGGREGATOR_QUEUE_DELAY 200
//Buffer capacity (in packets)
#define AGGREGATOR_BUFFER_CAPACITY  50
#define SOLVER_BUFFER_CAPACITY	15


///AggParams: Container for Aggregator thread parameters
struct AggParams : public Params{
  ///Maximum time allowed between packets sent at the same time from the same source
  Timestamp* time_toler; 
  Timestamp* time_to_wait;
  int num_sensors;
  
  virtual void print(ostream* fs);
  virtual void printcsv(ostream* fs);
  virtual AggParams* copy(void);
  virtual ParamsType getType(void) { return PT_AGG; };
};


/**Aggregator thread
 * 
 * This thread retrievs packets from the network
 * and when it's idle processes the packets it already has
 * 
 * Note: Socket events should not propagate past the AggThread.
 * 
 * @author Brian Shaw
 */
class AggThread : public wxThread{
protected:
  bool done;
  CondBuffer* buffer;
  CondBuffer* solverbuffer;
  ParamsWrapper* pwrap;
  PacketHeap* storageheap;
public:
  /// Called from main thread, this notifies aggregator of changes to the 
  /// user's configuration preferences. Expect this to be infrequent.
  //int notifyConfigChange(AggParams* newparams);
  
  
  // ----- Built-in functions for thread declared here --- //
  AggThread(CondBuffer* buf, CondBuffer* solverbuf, ParamsWrapper* apwrap);
  // thread execution starts here
  virtual void *Entry();
  void signalDone(void);
};

#endif
