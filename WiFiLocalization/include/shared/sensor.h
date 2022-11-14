/** sensor.h: Classes for making sensors appear on the screen
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
#ifndef FRAMEOBJECTS_H
#define FRAMEOBJECTS_H

#include <vector>

#include <wx/statbmp.h>
#include <wx/bmpbuttn.h>
#include <wx/socket.h>

#include "location.h"
#include "timestamp.h"
//Perhaps use one header with two different implementation files for the Frame?
//having dummy events and whatnot where they aren't needed??

using namespace std;


///Class for visually representing sensor nodes
class Sensor{
  friend class SensorList;
  friend class NRLFrame;
#if NRL_SWITCH_SIMULATOR
  friend class PacketGen;
#endif
protected:
  Location* loc;
  wxSocketBase* sock;
  Timestamp* time_initialized;
  Timestamp* most_recently_heard_from; //Solver only //NOTE: Do I need this?
  
  wxBitmapButton* button; //Needs to be accessible from NRLFrame, which #include's this header
  int id; //Associates this object with a specific button from the button pool
public:
  static wxBitmap* picture; //Global bitmap accessible through Sensor class
  Sensor(wxSocketBase* sock, int buttonid); //Solver //Other variants for different data structures?
  Sensor(Location* loc, int buttonid); //Simulator (which fills in the socket information later when simulator connects)
  ~Sensor(void);
  bool setLocation(Location* lcn); //Return value: Did it work? If it had a location, returns false
  bool setSocket(wxSocketBase* sock); //Return value: Did it work? If it had a socket, returns false
};


///A frame-wide list of all the sensors that exist
///This class provides more intuitive management of a vector of Sensors
class SensorList{
  friend class NRLFrame; //For simulator's sendall
protected:
  std::vector<Sensor*> slist;
public:
  int size(void) {return slist.size();}; //Returns number of sensors we have
  void append(wxSocketBase* sock, int id); //Create new sensor and append
  void append(Location* loc, int id); //Create new sensor and append
  void append(Sensor* sens);
  int remove(wxSocketBase* sock);
  int remove(int id);
  Sensor* getSensor(wxSocketBase* sock);
  Sensor* getSensor(int id);
  void PrintALL(void);
  vector<Location*> getLocations(void); //Return location of each sensor
};

#endif
