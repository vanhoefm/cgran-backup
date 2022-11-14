/** pktgen.h: Packet generator for the simulator
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
#ifndef NRL_PKTGEN_H
#define NRL_PKTGEN_H

using namespace std;

///FIXME: Error is not being introduced to the generated data!

#include "packet.h"
#include "simintruder.h"
#include "sensor.h"
#include "modeltemplates.h"


//FIXME: THis should come from parameters
//Speed of light in a vacuum in meters per nanosecond
#define C_NSEC	.299792458


///A wrapper for the packet generating functions
class PacketGen{
protected:
  PathLossModel* rssimodel;
  ToADistanceCalculator* toamodel;
public:
  Packet* genPacket(Intruder* src, Sensor* dest, Timestamp* time);
  
  PacketGen(PathLossModel* rmodel, ToADistanceCalculator* tmodel);
  ~PacketGen(void);
  /// Switch to a different path loss model 
  /// (NOT thread safe unless pointer assignment is atomic)
  void newModel(PathLossModel* rmodel, ToADistanceCalculator* tmodel);
};

//FIXME: needs params, setting of path loss model, etc

#endif
