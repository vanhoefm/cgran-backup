/** timestamp.h: A class for representing timestamps associated with packets
 * 
 * @author Brian Shaw
 * 
 * Portability note: stdint.h and sys/time.h are both required.
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

#ifndef TIMESTAMP_H
#define TIMESTAMP_H

#include <stdint.h>
#include "templates.h"

using namespace std;

//Constants
#define ONE_BILLION 1000000000


// Timestamps containing the date and time to sub-nanosecond precision.
class Timestamp : public Printable, public Serializable {
protected:
  int64_t thetime; //Year through second, see C++ Time
  double nsec; //8 bytes, 64 bits
  char timechar(char inchar, bool* lastcharvalid, bool* valid, bool* dp); //Helper function for timebuilder
public:
  //Unserialisation constructor
  Timestamp(char* buf, int bufsize);
  //Other constructors and functions
  Timestamp(double nsectoadd); //Makes a timestamp of "gettimeofday() + nsectoadd"
  Timestamp(void);
  Timestamp(int64_t secs, double nsecs);
  Timestamp(Timestamp* other);
  double getnsec (void);
  int64_t gettime (void); 
  //Required by base classes
  void print(ostream* fs);
  void printcsv(ostream* fs);
  Timestamp* copy(void);
  int serialize(char* buf, int buflen);
  //Functions for aggregator
  //Note: Some basic timestamp functions are also used there
  bool isExpiring(Timestamp* time_to_live, Timestamp* last_collection_time);
  
  bool isNear(Timestamp* other, Timestamp* tolerance);
  bool isSame(Timestamp* other);
  
  void normalize(void);
  double todouble(void); //Converts this timestamp to a double
  bool operator<(Timestamp other);
  Timestamp* operator+(Timestamp other);
  Timestamp* operator-(Timestamp other);
  
  //Get timestamps from Ryan's format
  bool timeBuilder(char* inbuf, int buflen);
};

// Function for determining how big a timestamp is

int getTimestampSize(void);

#endif