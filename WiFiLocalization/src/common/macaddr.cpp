/** macaddr.cpp: MAC address handling class
 * (used by Packet, Measurement, both Intruders)
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

#include <string.h>
#include <stdlib.h>
#include <iomanip>

#include "macaddr.h"
#include "bitutils.h"		

using namespace std;


unsigned char hex_to_octet(char highbyte, char lowbyte, bool* success){
  //Note: highbyte and lowbyte are interpreted as unsigned char
  unsigned char octet;
  *success = true;
  
  //cout << "Highbyte: " << (int) highbyte << endl;
  //cout << "Lowbyte: " << (int) lowbyte << endl;
  
  //Handle upper half of the byte
  if ((highbyte >= 48) && (highbyte <= 57)){
    //Number
    octet = (highbyte - 48) * 16;
  } else if ((highbyte >= 65) && (highbyte <= 70)){
    //Capital letter
    octet = (highbyte - 55) * 16;
  } else if ((highbyte >= 97) && (highbyte <= 102)){
    //Lowercase letter
    octet = (highbyte - 87) * 16;
  } else {
    octet = 0;
    *success = false;
  }
  int q = octet;
  
  //Handle lower half of the byte
  if (lowbyte >= 48 && lowbyte <= 57){
    //Number
    octet += (lowbyte - 48);
  } else if ((lowbyte >= 65) && (lowbyte <= 70)){
    //Capital letter
    octet += (lowbyte - 55);
  } else if ((lowbyte >= 97) && (lowbyte <= 102)){
    //Lowercase letter
    octet += (lowbyte - 87);
  } else if (lowbyte == -1){
    //Not using low byte
  } else {
    *success = false;
  }
  
  return octet;
}


MacAddr::MacAddr(){
}

bool MacAddr::macBuilder(const char* macaddr, int macsize){
  return macBuilder((char*) macaddr, macsize);
}

bool MacAddr::macBuilder(unsigned char* macaddr, int macsize){
  return macBuilder((char*) macaddr, macsize);
}

bool MacAddr::macBuilder(uint64_t macaddr){
  addr = macaddr;
  return (macaddr > 0xFFFFFFFFFFFFull);
}


int MacAddr::serialize(char* buf, int buflen){
  if (buflen < sizeof(uint64_t))
    throw "Can't serialize MAC address, not enough buffer space";
  
  nrl_htonul(addr, buf);
  return (int) sizeof(uint64_t);
}
  
MacAddr::MacAddr(char* buf, int buflen){
  if (buflen < sizeof(int64_t))
    throw "Can't serialize MAC address, not enough buffer space";
  
  addr = nrl_ntohul(buf);
}

MacAddr* MacAddr::copy(void){
  MacAddr* mac = new MacAddr();
  mac->macBuilder(addr); //Always returns TRUE unless we are invalid
  return mac;
}

void MacAddr::print(ostream* fs){
  *fs << "MAC address: ";
  printcsv(fs);
  *fs << endl;
}

void MacAddr::printcsv(ostream* fs){
  unsigned int octet[6];
  int a;
  uint64_t mask = 0xFF0000000000ull;
  for (a = 0; a < 6; a++){
    octet[a] = (addr & mask) >> (8 * (5 - a));
    mask = mask >> 8;
  }
  *fs << hex << setfill('0');
  for (a = 0; a < 5; a++)
    *fs << setw(2) << octet[a] << ":";
  *fs << setw(2) << octet[5] << dec << setfill(' ');
}


bool MacAddr::sameMac(MacAddr* other){
  if (other == NULL)
    return false;
  else
    return addr == other->addr;
}


//Convert hack address (if used) and AA:BB:CC:DD:EE:FF into octets
//Note: char* is quietly typecast to unsigned char* output
bool MacAddr::macBuilder(char* macaddr, int macsize){
  int a;
  unsigned char octet[6];
  bool validflag = true;
  
  if (macsize == 6){
    //Packet represents octet format
    for (a = 0; a < 6; a++)
      octet[a] = (unsigned char) macaddr[a];
  } else if (macsize == 12){
    //FFEEDDCCBBAA mac (ASCII text, no delimiters)
    for (a = 0; a < 6; a++)
      octet[a] = hex_to_octet(macaddr[2*a], macaddr[2*a+1], &validflag);
  } else if (macsize == 17){
    //FF:EE:DD:CC:BB:AA format (the delimiters are not checked; they can be any character)
    for (a = 0; a < 6; a++)
      octet[a] = hex_to_octet(macaddr[2*a+a], macaddr[2*a+a+1], &validflag);
  } else {
    cout << "Warning: Invalid macsize" << endl;
    validflag = false;
  }
  
  if (validflag){
    //Create MAC address
    addr = 0;
    for (a = 0; a < 6; a++){
      uint64_t temp = octet[a];
      addr += temp << 8 * (5 - a); //Bit shift
    }
  }
  
  return validflag;
}
