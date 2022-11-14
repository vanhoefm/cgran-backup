/** timestamp.cpp: Implementation of Timestamp class functions
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

#include "timestamp.h"
#include "bitutils.h"

#include <sys/time.h>
#include <math.h>
#include <iomanip> //for setting precision while printing
#include <string.h>
#include <stdlib.h>

//For perror
#include <stdio.h>
#include <errno.h>

using namespace std;

/*--------------Implementations of Timestamp class-------------------*/

int getTimestampSize(void) {
  return (int) (sizeof(int) + sizeof(double));
}

double Timestamp::getnsec (void) {
  return nsec;
}

int64_t Timestamp::gettime (void) {
  return thetime;
}

double Timestamp::todouble(void){
  if (thetime > 4500000) //2^52 / 1 billion (rounded down)
    cout << "Warning! Timestamp conversion to double overflowed!" << endl;
  return ((double) thetime) * ONE_BILLION + nsec;
}

void Timestamp::print(ostream* fs){
  int defaultprecision = fs->precision();
  *fs << setprecision(10);
  normalize();
  *fs << thetime << " seconds and " << nsec << " nanoseconds\n";
  *fs << setprecision(defaultprecision);
}

void Timestamp::printcsv(ostream* fs){
  int defaultprecision = fs->precision();
  *fs << setprecision(10);
  *fs << thetime << "," << nsec;
  *fs << setprecision(defaultprecision);
}

Timestamp* Timestamp::copy(void){
  return new Timestamp(*this);
}


//Constructor for making microsecond-accurate timestamps with gettimeofday()
//NOTE gettimeofday is microsecond accurate
//if you just want current time, pass in 0
Timestamp::Timestamp(double nsectoadd){
  /* gettimeofday timestamps
  struct timeval time;
  gettimeofday(&time, NULL);
  thetime = (int) time.tv_sec;
  nsec = ((double) time.tv_usec) * 1000 + nsectoadd;
  */
  //ns precision method
  struct timespec time;
  int err = clock_gettime(CLOCK_REALTIME, &time);
  if (err){
    perror("Error getting system clock: ");
    throw "Timestamp error";
  }
  thetime = time.tv_sec;
  nsec = (double) time.tv_nsec + nsectoadd;
}

Timestamp::Timestamp(void){
  /* gettimeofday method
  struct timeval time;
  gettimeofday(&time, NULL);
  thetime = (int) time.tv_sec;
  nsec = ((double) time.tv_usec) * 1000;
  */
  struct timespec time;
  int err = clock_gettime(CLOCK_REALTIME, &time);
  if (err){
    perror("Error getting system clock: ");
    throw "Timestamp error";
  }
  thetime = time.tv_sec;
  nsec = (double) time.tv_nsec;
}

/*
Timestamp::Timestamp(struct timeval tv){
  thetime = tv.tv_sec;
  nsec = tv.tv_usec * 1000;
}
*/

Timestamp::Timestamp(Timestamp* other){
  thetime = other->gettime();
  nsec = other->getnsec();
}


Timestamp::Timestamp(int64_t secs, double nsecs){
  thetime = secs;
  nsec = nsecs;
}


int Timestamp::serialize(char* buf, int bufsize){
  //Note: Timestamp serialization forces 64-bit time_t
  if (bufsize < sizeof(int64_t) + sizeof(double))
    throw "ERROR: Buffer is not big enough to store timestamp.";
  int a = 0;
  memset(buf, '\0', bufsize);
  nrl_htonul(thetime, &buf[a]);
  a += sizeof(int64_t);
  nrl_htond(nsec, &buf[a]);
  a += sizeof(double);
  return a;
}


//Unserialization constructor
Timestamp::Timestamp(char* buf, int bufsize){
  //Note: Timestamp serialization forces 64-bit time_t
  if (bufsize < sizeof(double) + sizeof(int64_t))
    throw "Buffer is not big enough to read timestamp from.";  
  int a = 0;
  thetime = nrl_ntohul(&buf[a]);
  a += sizeof(int64_t);
  nsec = nrl_ntohd(&buf[a]);
  return;
}


bool Timestamp::isNear(Timestamp* other, Timestamp* tolerance){
  Timestamp* min = *other - *tolerance;
  Timestamp* max = *other + *tolerance;
  bool retval = (*min < *this) && (*this < *max);
  delete min;
  delete max;
  return retval;
}


bool Timestamp::isSame(Timestamp* other){
  return (thetime == other->thetime) && (nsec == other->nsec);
}


//Check to see if this timestamp is expiring
//(aggregator function)
bool Timestamp::isExpiring(Timestamp* time_to_live, Timestamp* last_collection_time){
  Timestamp* ts = (*last_collection_time - *time_to_live);
  bool retval;
  retval = *this < *ts;
  delete ts;
  return retval;
}


//Fix this timestamp so that the nsec is not greater than one second
//and so there are no negative values. Performance matters here.
void Timestamp::normalize(void){
  double val = floor(nsec / ONE_BILLION);
  thetime = thetime + val;
  nsec = nsec - val * ONE_BILLION;
}


//Note: The two timestamps being added should NOT be pointers!
///@return: The sum of the two timestamps, allocated on the heap
//The sum is then normalized so the nsec is not too big.
Timestamp* Timestamp::operator+(Timestamp other){
  Timestamp* ts = new Timestamp(other);
  ts->thetime = thetime + floor((ts->nsec + nsec) / ONE_BILLION);
  ts->nsec = fmod(ts->nsec + nsec, ONE_BILLION);
  ts->normalize();
  return ts;
}


Timestamp* Timestamp::operator-(Timestamp other){
  Timestamp* ts = new Timestamp(other);
  ts->thetime = thetime - other.thetime;
  ts->nsec = nsec - other.nsec;
  ts->normalize();
  return ts;
}


bool Timestamp::operator<(Timestamp other){
  other.normalize();
  normalize();
  if (thetime < other.thetime)
    return true;
  if ((thetime == other.thetime) && (nsec < other.nsec))
    return true;
  return false;
}

///NOTE: timechar and timebuilder are untested. The previous method works, but crashes on corrupt packets.

//Helper function for timeBuilder
char Timestamp::timechar(char inchar, bool* lastcharvalid, bool* valid, bool* dp){
  *dp = false;
  if (*valid){
    if ((48 <= inchar) && (inchar <= 57)){
      //Valid number
      if (! *lastcharvalid)
	*valid = false; //If we got an invalid character before this valid number, the timestamp has an error.
      *lastcharvalid = true;
      return inchar;
    } else if (inchar == 46) {
      //'.' arrived
      *dp = true;
      return '\0';
    } else {
      //Not a number or decimal point
      cout << "Not a number or decimal" << endl;
      *lastcharvalid = false;
      return '\0';
    }
  }
}


//Ryan's pkt format decoder
//Expects a timestamp in ASCII, in the form "SSSSSS.NNNNNNN" (unknown length, but a \0 before the end of the array)
//Call this after making a timestamp through the default constructor
//Return value: true for valid timestamp, false for known invalid data.
bool Timestamp::timeBuilder(char* inbuf, int buflen){
  if (buflen >= 31){
    //Timestamp won't fit in internal buffer (this should NEVER be an issue)
    throw "timeBuilder internal buffer isn't big enough for your timestamp!";
  }
  bool lastcharvalid, valid, dp;
  int a, b;
  char buf[32];
  memset(buf, '\0', 32);
  b = 0;
  valid = true;
  lastcharvalid = true;
  for (a = 0; a < buflen; a++){
    buf[b] = timechar(inbuf[a], &lastcharvalid, &valid, &dp);
    if (dp && valid){
      //Decimal point recv'd, store the time in seconds
      cout << "Decimal point and 'valid'" << endl;
      if (a != b)
	valid = false; //This condition only holds if a decimal point was found before.
      buf[b] = '\0';
      thetime = atoll(buf);
      cout << "Thetime is " << thetime << endl;
      memset(buf, '\0', 32);
      b = -1;
    }
    b++;
  }
  //Loop is finished. thetime should have been filled when the decimal point was reached.
  //if it wasn't, no decimal point appeared and this means a == b. 
  //Decimal points should never appear as the first character, so data was corrupted.
  if (a == b)
    valid = false;
  //Use buffer to fill nsec.
  if (valid){
    cout << "Valid timestamp (supposedly)" << endl;
    buf[9] = '\0';
    cout << "nsec will be " << atoll(buf) << endl;
    cout << "nsec offset by " << 10 * (9 - strlen(buf)) << endl;
    nsec = atoll(buf);// + 10 * (9 - strlen(buf)); //The second term compensates for truncated decimal places.
  }
  return valid;
}


