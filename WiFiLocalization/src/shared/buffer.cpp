/** condbuffer.cpp: Implementation of condition-based buffer
 * 
 * It's a buffer for passing pointers between threads, with blocking add/remove calls.
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

#include "buffer.h"

#include <iostream>

/*-----------------Implementation of Buffer--------------------*/

CondBuffer::CondBuffer(int cpy){
  needstoexit = false;
  //Capacity sets an upper limit on how many items can be in the buffer without warnings appearing
  //This is to prevent the solver from being overloaded, not due to limitations in std::deque
  capacity = cpy;
  if (cpy <= 0)
    capacity = BUFFER_DEFAULT_CAPACITY;
  
  //Make a mutex and maybe some condition variables
  //lock = new wxMutex();
  bufferready = new wxCondition(lock);
}


CondBuffer::~CondBuffer(void){
  //NOTE: This destructor assumes all producers and consumers are no longer using the buffer!
  //It will not check for this or notify any threads!
  
  //Destructor will remove all packets from the buffer!
  //Warning: This does not check to see if any other thread is using the buffer!
  //It is wise to kill the threads that need this buffer first (especially the consumer thread)
  
  //Check to see if anything is in the buffer before destroying it
#if CHECK_BUFFER_CONTENTS_WHEN_FLUSHING
  int a;
  for (a = 0; a < pktarray.size(); a++){
    if (pktarray[a] != NULL) {
      printf("There was an item in the buffer when flushed! Unable to delete\n");
      //delete pktarray[a]; //may cause segfaults if it is not set to NULL when items are removed
    }
  }
#endif

  //Free up mutex and condition variables
  delete bufferready;
} 


//Note: In this current implementation, the mutex is acquired twice
//if that becomes a problem, come up with a private version of is_full
//that acquires the mutex but does not release it unless there is an error
int CondBuffer::addone(void* pkt){
  //printf("Adding packet! or trying...?\n");
  
  if (pkt == NULL){
    printf("Warning! Trying to add null packet to buffer\n");
    return 1;
  }
  
  wxMutexError err = lock.Lock(); //Note: wxMutexLocker is not used since Signal() unlocks the mutex
  if (err == wxMUTEX_DEAD_LOCK)
    throw "CondBuffer addone mutex deadlock\n";
  
  // CRITICAL SECTION! The mutex is released when function returns.
  
  if (pktarray.size() >= capacity){
    printf("Warning! Extending buffer's capacity\n");
  }
  
  pktarray.push_back(pkt);
  
  bufferready->Signal();
  err = lock.Unlock(); //Note: wxMutexLocker is not used since Signal() unlocks the mutex
  if (err == wxMUTEX_DEAD_LOCK)
    throw "CondBuffer add mutex deadlock 2\n";
  
  return 0;
}


int CondBuffer::add(std::vector<void*> pkts){
  
  //Check to see if we actually have items to add.
  if (pkts.size() <= 0)
    return -1;
  
  int a;
  
  wxMutexError err = lock.Lock(); //Note: wxMutexLocker is not used since Signal() unlocks the mutex
  if (err == wxMUTEX_DEAD_LOCK)
    throw "CondBuffer add mutex deadlock\n";
  
  // CRITICAL SECTION! The mutex is released when Signal() is called.
    
  for (a = 0; a < pkts.size(); a++){
    void* pkt = pkts[a];
    if (pkt == NULL){
      printf("Warning! Trying to add null packet 2\n");
      //return 1;
    } else {
      if (pktarray.size() >= capacity)
	printf("Warning! Extending buffer's capacity 2\n");
      pktarray.push_back(pkt);
    }
  }
  
  //Let the consumer know that the buffer has items in it.
  bufferready->Signal();
  err = lock.Unlock(); //Note: wxMutexLocker is not used since Signal() unlocks the mutex
  if (err == wxMUTEX_DEAD_LOCK)
    throw "CondBuffer add mutex deadlock 2\n";
  
  return 0;
}


//Retrieve and remove a packet from the buffer
//Called by the consumer thread
///@arg timeout
///@arg mythread: Thread object pointer for calling TestDestroy() if needed, void otherwise
void* CondBuffer::removeone(int timeout, wxThread* mythread, bool allowemptyreturns){
  
  wxMutexError err = lock.Lock(); //Note: wxMutexLocker is not used since Signal() unlocks the mutex
  if (err == wxMUTEX_DEAD_LOCK)
    throw "CondBuffer removeone mutex deadlock\n";
  
  // CRITICAL SECTION! It is now OK to check the buffer
  
  if (pktarray.size() == 0 && mythread == NULL)
    return NULL; //Main thread is a GUI and should never block
    
  while (pktarray.size() <= 0){
    
    bufferready->WaitTimeout(timeout); //Loses and re-acquires mutex
    
    if (mythread != NULL)
      needstoexit = needstoexit || mythread->TestDestroy();
    
    if (needstoexit)
      return NULL;
    
    if (allowemptyreturns)
      break;
  }
  
  void* tmppkt = NULL;
  if (pktarray.size() > 0){
    //Grab oldest packet pointer from the front of the array
    tmppkt = pktarray[0];
    pktarray[0] = NULL;
    pktarray.pop_front();
  }
  
  bufferready->Signal();
  err = lock.Unlock(); //Note: wxMutexLocker is not used since Signal() unlocks the mutex
  if (err == wxMUTEX_DEAD_LOCK)
    throw "CondBuffer removeone mutex deadlock 2\n";
  
  return tmppkt;
}


std::deque<void*>* CondBuffer::remove(int timeout, wxThread* mythread, bool allowemptyreturns){
  
  if (needstoexit){
    return NULL;
  }
  
  if (pktarray.size() == 0 && mythread == NULL){
    return NULL; //Main thread is a GUI and should never block  
  }
  
  wxMutexError err = lock.Lock(); //Note: wxMutexLocker is not used since Signal() unlocks the mutex
  if (err == wxMUTEX_DEAD_LOCK)
    throw "CondBuffer remove mutex deadlock\n";
  
  // CRITICAL SECTION! It is now OK to check the buffer
  while (pktarray.size() <= 0){
    
    bufferready->WaitTimeout(timeout); //Loses and re-acquires mutex
    
    if (mythread != NULL)
      needstoexit = needstoexit || mythread->TestDestroy();
    
    if (needstoexit){
      lock.Unlock();
      return NULL;
    }
    
    if (allowemptyreturns)
      break;
  }
  
  std::deque<void*>* retarray = new std::deque<void*>(pktarray);
  pktarray.clear();
  
  bufferready->Signal();
  err = lock.Unlock(); //Note: wxMutexLocker is not used since Signal() unlocks the mutex
  if (err == wxMUTEX_DEAD_LOCK)
    throw "CondBuffer remove mutex deadlock 2\n";
  
  return retarray;
}


std::deque<void*>* CondBuffer::pollremove(void){
  if (needstoexit)
    return NULL;
  
  wxMutexError err = lock.Lock();
  if (err == wxMUTEX_DEAD_LOCK)
    throw "CondBuffer remove mutex deadlock\n";
  
  //CRITICAL SECTION
  std::deque<void*>* retarray = new std::deque<void*>(pktarray);
  pktarray.clear();
  
  bufferready->Signal();
  err = lock.Unlock();
  if (err == wxMUTEX_DEAD_LOCK)
    throw "CondBuffer remove mutex deadlock 2\n";
  
  return retarray;
}


void CondBuffer::signalDone(void){
  needstoexit = true;
  bufferready->Broadcast();
}
