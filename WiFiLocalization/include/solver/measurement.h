/** measurement.h: Class for "measurements"
 * Measurements are groups of packets that have been correlated
 * and represent a single transmitted intruder packet as measured by all of the sensors
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

#ifndef MEASUREMENT_H
#define MEASUREMENT_H

#include <vector>

#include "templates.h"
#include "packet.h"
#include "macaddr.h"

using namespace std;

//It would not be hard to make this serializable, but that feature is not needed.
class Measurement: public Printable{
protected:
  std::vector<Packet*>* pktarray;
  MacAddr* mac;
  Timestamp* timedetected;
  Location* position;
  Location* uncertainty;
  
public:
  Measurement();
  ~Measurement();
  bool measbuilder(std::vector<Packet*>*); //Return value: if this is a valid measurement.
  
  //Add in other Measurement functions from older solver versions
  //keeping in mind that Measurements still need to be used as updates for Intruders
  Location getPosition(void);
  Location* getPositionHeap(void);
  Location getUncertainty(void);
  MacAddr* getMac(void); ///See note in .cpp file
  Timestamp* getDetectionTime(void);
  bool sameMac(MacAddr* intrudermac);
  
  //Required by Printable
  void print(ostream* fs);
  void printcsv(ostream* fs);
  Measurement* copy(void);
};


#endif
