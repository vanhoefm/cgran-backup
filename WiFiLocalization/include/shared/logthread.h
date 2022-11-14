/** logthread.h: Logger thread and wrapper class
 * 
 * Pass Printable items to the logger thread and it will
 * asynchronously write them to disk
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
#ifndef LOGTHREAD_H
#define LOGTHREAD_H

#include "buffer.h"
#include "logger.h"

using namespace std;


class LogThread : public wxThread{
private:
  bool done;
protected:
  NRLLogger mylogger;
  CondBuffer* buffer; //Note: Don't delete this, the main thread owns it!
public:
  // ----- Solver-specific functions declared here --- //
  void log(Printable* item, LogItemType type); //Called by producers!
  // ----- Built-in functions for thread declared here --- //
  LogThread(CondBuffer* logbuf);
  // thread execution starts here
  virtual void *Entry();
  //Function for waking up this thread when it's time to kill it.
  void signalDone(void);
};


//Wrapper for wxThreads interface (Logger thread)
//It deletes the buffer! Therefore this cannot be deleted until all other threads
//using the buffer are stopped
class TheLoggerThread{
protected:
  LogThread* logthread;
  
public:
  CondBuffer* buffer;
  
  TheLoggerThread(void);
  ~TheLoggerThread(void);
  void log(Printable* item, LogItemType type); //wrapper for thread's log() function
};



#endif
