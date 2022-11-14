/** logger.cpp: Two classes for writing information to the appropriate logfile
 * 
 * "Logger" is a wrapper around each logfile handling the file stream and actual I/O
 * "NRLLogger" keeps track of all the logfiles and sorts arriving information into each
 * Both are called from the logging thread
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
///Logger thread side logging functions!

#include "templates.h"
#include "buffer.h"
#include "logger.h"

using namespace std;


/*-----------Logger implementation--------*/

Logger::Logger(char* logfilename){
  fs.open(logfilename, ios::out | ios::app); // | ios::trunc);
  if (!fs.is_open())
    throw "Unable to open log file\n";
}

//BUG: Files are not able to be truncated successfully (if this is enabled it can't open the files)

Logger::Logger(const char* logfilename){
  fs.open(logfilename, ios::out | ios::app); // | ios::trunc);
  if (!fs.is_open())
    throw "Unable to open log file\n";
}

Logger::~Logger(void){
  fs.close();
}

void Logger::log(Printable* item){
  item->print(&fs);
  return;
}

void Logger::flush(void){
  fs.flush();
}


/*-------NRLLogger implementation--------*/

NRLLogger::NRLLogger(void){ //can be created on the stack if this remains NRLLogger(void)
  loglist = new std::vector<Logger*>;
#if NRL_SWITCH_SOLVER
  loglist->push_back(new Logger("solver_warnings.txt"));
  loglist->push_back(new Logger("solver_packets.txt"));
#else
  loglist->push_back(new Logger("simulator_warnings.txt"));
  loglist->push_back(new Logger("simulator_packets.txt"));
#endif
}


NRLLogger::~NRLLogger(void){
  //Delete all loggers, then the list
  int a;
  for (a = 0; a < NUM_LOGGERS; a++)
    delete (*loglist)[a];
  delete loglist;
}


void NRLLogger::log(Printable* item, LogItemType type){
  
  Logger* logfile;
  Printable* item2 = NULL; //In case we need to log a warning message
  
  //Each possible logger destination
  switch (type){
    
    case LT_RECV_PKT:
      logfile = (*loglist)[1];
      break;
    
    default: 
      //Send a warning to the "warnings" logfile
      logfile = (*loglist)[0];
      item2 = new NRLString("Warning! This object's logfile was not specified.\n");
      break;
  }
  
  //Log any warnings
  if (item2 != NULL)
    logfile->log(item2);
  //Log the item
  logfile->log(item);
  //NOTE: This costs some logger thread performance in favor of
  //increased protection against crashes and ease of debugging
  return;
}

  
/*----------NRLString implementation------------*/

NRLString::NRLString(char* buf, size_t len){
  str.assign((const char*) buf, len);
}

NRLString::NRLString(const char* text){
  str.assign(text);
}

void NRLString::print(ostream* fs){
  *fs << str;
}

void NRLString::printcsv(ostream* fs){
  *fs << str;
}


NRLString* NRLString::copy(void){
  return new NRLString(*this); //Built in copy constructor
}




