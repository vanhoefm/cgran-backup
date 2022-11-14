/** macaddr.h: MAC address handling class
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

#ifndef NRL_MACADDR_H
#define NRL_MACADDR_H

#include <stdint.h>
#include "templates.h"

#define USE_HACK_ADDRESS 0

using namespace std;

unsigned char hex_to_octet(char highbyte, char lowbyte, bool* success);

///MAC address should always be converted into and out of big-endian
///as this is the format both humans and networks use


class MacAddr : public Serializable, public Printable{
  uint64_t addr;
public:
  ///macBuilder::fills in the address of a MAC address
  ///@return: True for success, false for failure
  bool macBuilder(char* macaddr, int macsize);
  bool macBuilder(const char* macaddr, int macsize);
  bool macBuilder(unsigned char* macaddr, int macsize);
  bool macBuilder(uint64_t macaddr);
  MacAddr();
  bool sameMac(MacAddr* other);
  //Needed to fit the templates
  MacAddr* copy(void);
  void print(ostream* fs);
  void printcsv(ostream* fs);
  int serialize(char* buf, int buflen);
  MacAddr(char* buf, int buflen);
};

#endif