/** intruder.h: Classes for tracking and displaying the locations of wireless transmitters
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

#ifndef NRL_INTRUDERS_H
#define NRL_INTRUDERS_H

#include <deque>
#include <vector>

#include "nrlintruder.h"
#include "switch.h" //Tells us whether this is the solver or the simulator.
#include "timestamp.h"
#include "location.h"
#include "app.h"
#include "intruderpattern.h"

using namespace std;


class Intruder : public NRLIntruder{
  friend class PacketGen;
private:
  IntruderPattern* mypattern;
public:
  Intruder(MacAddr* macaddr, Location* loc, IntruderPattern* pattern, NRLApp* app);
  ~Intruder(void);
  bool update(void); //Return value: if the intruder should be deleted
};


///A class that the Frame uses to manage its Intruders
class TheIntruders : public NRLTheIntruders{
  friend class NRLFrame; //for sendAllPackets()
private:
  std::vector<Intruder*> intruders;
  
public:
  TheIntruders(NRLApp* app);
  int size(void) { return intruders.size(); };
  void add(Intruder* intruder);
  void add(std::vector<Intruder*> intruderlist);
  std::vector<int> updateAll(void); //Changes velocity if one is provided, returns IDs of deleted Intruders
};

#endif
