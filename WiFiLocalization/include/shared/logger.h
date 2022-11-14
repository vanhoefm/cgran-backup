/** logger.h: Two classes for writing information to the appropriate logfile
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
#ifndef LOGGER_H
#define LOGGER_H

#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <errno.h>

#include <wx/thread.h>

#include <fstream>
#include <vector>

#include "templates.h"
#include "buffer.h"

using namespace std;

#define NUM_LOGGERS	2
#define BUF_ARRAY_SIZE	10
#define UPDATE_RATE	100

//How long logger should wait between checks for packets
#define LOGGER_QUEUE_DELAY	200
#define LOGGER_BUFFER_CAPACITY	20


///LogItemType: Types of items that can be sent to the logger
///Used to specify which logfile gets the data being logged
///Default: Send to the logfile of warnings and orphaned information
enum LogItemType { LT_RECV_PKT, LT_OTHER };


///NRLString: A wrapper for std::string that is Printable
///(therefore able to be passed to the logger)
class NRLString : public string, public Printable{
protected:
  string str;
public:
  NRLString(char* buf, size_t size);
  NRLString(const char* text);
  void print(ostream* fs);
  void printcsv(ostream* fs);
  NRLString* copy(void);
};


///Generic class for all loggers
class Logger{
private:
  ofstream fs;
  
public:
  //Note: Both constructors do the same thing
  Logger(char* logfilename);
  Logger(const char* logfilename);
  ~Logger(void);
  void log(Printable* item);
  void log(Printable* item, bool blocking);
  void flush(void); //Force it to write buffer (and a newline) to file
};


///Global object for handling the logging
class NRLLogger{
protected:
  std::vector<Logger*>* loglist;
  
public:
  NRLLogger(void);
  ~NRLLogger(void);
  
  void log(Printable* item, LogItemType type);
  //Multiple packet log?
};


/*----------Typedefs------------------*/

/*----------Structures----------------*/

//Container for Printables and information for deciding where they get logged to
struct LoggerItem{
  Printable* item;
  LogItemType type;
};

/*----------Global Variables-----------*/

/*---------Function prototypes---------*/

#endif
