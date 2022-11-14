#ifndef BUFFER_H
#define BUFFER_H

/** condbuffer.h: Implementation of condition-based buffer
 * 
 * It's a buffer for passing pointers between threads, with blocking add/remove calls.
 * 
 * The producer thread(s) may try to insert items into the buffer at any time.
 * A blocking call to the buffer's mutex occurs, then the item is inserted and a sleeping
 * consumer thread is woken up. 
 * The consumer thread will first acquire the mutex and see if anything is present.
 * If not, it will go to sleep, checking periodically to see if the thread should be joined.
 * After it is awake, it checks the buffer's size and either sleeps or eats the data present.
 * 
 * The producer never sleeps. The consumer always checks to see if anything is available before
 * sleeping, therefore any signals that occur during processing can be safely missed.
 * 
 * This class is intended for a one-producer/one-consumer interface
 * or a several-producer/one-consumer interface. Other uses should work but may not be optimal.
 * 
 * The underlying buffer is a FIFO std::deque (double-ended queue)
 * 
 * This class is used for the logging feature (many producers one consumer)
 * and for passing packet/frame data between the GUI/event thread and processing threads
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

#include <wx/thread.h>
#include <vector>
#include <deque>

#define CHECK_BUFFER_CONTENTS_WHEN_FLUSHING 	1
#define BUFFER_DEFAULT_CAPACITY	10

using namespace std;

//Generic buffer class for item passing between 
//several writing thread and 1 reading thread
//This uses wxWidgets' condition variables instead of a mutex.
//FIXME: Make this a CondBuffer<Printable*> to avoid C-style typecasting
class CondBuffer{
private:
  wxMutex lock;
protected:
  int capacity;
  std::deque<void*> pktarray;
  /** The buffer signals one thread when it has been updated
   * but if it is full, it signals all of them.
   * This lets it be used efficiently: Multiple threads can add to it one after the other,
   * but they are expected to yield to the consumer if they find it's full.
   * (They are all signaled when the buffer is about to be deleted, as well)
   */
  wxCondition* bufferready; 
  bool needstoexit; //"Atomic" boolean used as a supplement to TestDestroy
public:
  CondBuffer(int cpy); //Capacity is required, 0 indicates "use default"
  ~CondBuffer(void); //Destructor will remove all items from the buffer!
  
  int addone(void* pkt);
  int add(std::vector<void*> pkts);
  //Consumer threads should pass "this" so it can call TestDestroy, main threads should pass NULL.
  //If TestDestroy returns TRUE or NULL was passed, the buffer check will not wait for the signal
  //allowemptyreturns determines if the remove() function is allowed to return 0 items.
  //If false, it will block again until items or a "done" signal are received.
  void* removeone(int timeout, wxThread* mythread, bool allowemptyreturns); 
  std::deque<void*>* remove(int timeout, wxThread* mythread, bool allowemptyreturns);
  
  std::deque<void*>* pollremove(void); //Blocking Remove all items currently in buffer (for use as ordinary buffer)
  //Workaround for getting everything to terminate cleanly
  void signalDone(void);
};

#endif
