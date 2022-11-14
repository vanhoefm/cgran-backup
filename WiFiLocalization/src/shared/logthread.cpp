/** logthread.cpp: Logger thread and wrapper class
 * 
 * Pass Printable items to the logger thread and it will write them to disk for you
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
using namespace std;


#include "logthread.h"


/*----------LogThread implementation----------*/

// Thread constructor used to pass in and store arguments 
// (only one is needed)
LogThread::LogThread(CondBuffer* logbuf) : wxThread(wxTHREAD_JOINABLE){
  if (logbuf == NULL)
    throw "Cannot run logger thread without initializing the output streams\n";
  
  done = false;
  buffer = logbuf;
}

//Helper thread to manage all loggers
/// Thread execution function
void* LogThread::Entry(){
    
  //Set up our NRLLogger on the stack
  NRLLogger mylogger;
  
  cout << "INFO: Logger thread running\n";
  
  int a;
  std::deque<void*>* items;
  
  while (!done){
    
    //cout << "Logger Attempting remove" << endl;
    //Grab everything in the buffer
    items = buffer->remove(LOGGER_QUEUE_DELAY, This(), false); //Blocks when buffer is empty
    
    //cout << "Logger Remove succeeded" << endl;
    
    if (items != NULL){ //NULL gets returned when it's time to exit
      //Log the items
      for (a = 0; a < items->size(); a++){
	LoggerItem* item = (LoggerItem*) (*items)[a];
	mylogger.log(item->item, item->type);
	delete items;
	delete item; //Deletes the struct but keeps the contents alive!
      }
    }
    
    //cout << "Calling TestDestroy" << endl;
    done = done || TestDestroy(); //REQUIRED: Check to see if the thread needs to exit   
  }
  
  //Exit and join with main thread
  return NULL;
}


//MAIN LOGGER INTERFACE FOR THE REST OF THE CODE!!
//NOTE: This creates a copy of the item being logged.
//The copy is owned by the logger, so the original can be safely deleted.
void LogThread::log(Printable* item, LogItemType type){
  //Build container struct
  LoggerItem* logitem = new LoggerItem;
  logitem->item = item->copy();
  logitem->type = type;
  //Place container into buffer (may block on a mutex)
  cout << "Adding logger item to the buffer" << endl;
  buffer->addone((void*) logitem);
}


void LogThread::signalDone(void){
  done = true;
  buffer->signalDone();
}


/*------TheLoggerThread implementation-----*/

//Constructor. Create logger thread if OK, otherwise throw error.
TheLoggerThread::TheLoggerThread(void){
  
  //Make buffer
  buffer = new CondBuffer(LOGGER_BUFFER_CAPACITY); 
//Ownership of the buffer is kept by the main thread so other threads will not be waiting on a nonexistent buffer
  
  //Create thread
  logthread = new LogThread(buffer);
  wxThreadError err = logthread->Create();
  if (err != wxTHREAD_NO_ERROR)
    throw "Unable to create logger thread\n";
  err = logthread->Run();
  if (err != wxTHREAD_NO_ERROR)
    throw "Unable to run logger thread\n";
}


TheLoggerThread::~TheLoggerThread(void){ 
  logthread->signalDone();  
  logthread->Wait();
  delete logthread;
  //cout << "About to delete buffer\n";
  delete buffer;
}


void TheLoggerThread::log(Printable* item, LogItemType type){
  try {
    cout << "About to logthread->log" << endl;
    logthread->log(item, type);
  } catch (const char* text) {
    cout << "Error in logging an item: " << text << endl;
  }
}
