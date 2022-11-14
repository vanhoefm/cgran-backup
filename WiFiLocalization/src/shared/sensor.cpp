/** sensor.cpp: Helper classes for making items appear on the screen
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
#include "sensor.h"

using namespace std;

wxBitmap* Sensor::picture = NULL;


Sensor::Sensor(wxSocketBase* socket, int buttonid){
  sock = socket;
  time_initialized = new Timestamp((double) 0); 
  loc = NULL;
  id = buttonid;
}

Sensor::Sensor(Location* lcn, int buttonid){
  loc = lcn;
  time_initialized = new Timestamp((double) 0); 
  sock = NULL;
  id = buttonid;
}


//Note: This function takes ownership of the location passed in
bool Sensor::setLocation(Location* lcn){
  bool isnull = (loc == NULL);
  if (!isnull)
    loc = lcn;
  return isnull;
}


//Note: This function takes ownership of the location passed in
bool Sensor::setSocket(wxSocketBase* socket){
  bool isnull = (socket == NULL);
  if (!isnull)
    sock = socket;
  else
    cout << "Warning! Attempted to set sensor's socket to NULL" << endl;
  return isnull;
}


Sensor::~Sensor(void){
  delete loc;
  delete time_initialized;
}


/*-----------SensorList implementation---------*/

void SensorList::append(wxSocketBase* sock, int id){
  Sensor* sens = new Sensor(sock, id);
  slist.push_back(sens);
}

void SensorList::append(Location* loc, int id){
  Sensor* sens = new Sensor(loc, id);
  slist.push_back(sens);
}

void SensorList::append(Sensor* sens){
  slist.push_back(sens);
}


//O(N) where N is number of sensors
///@return: ID previously used by the button
int SensorList::remove(wxSocketBase* sock){
  int a;
  for (a = 0; a < slist.size(); a++){
    if (slist[a]->sock == sock){
      int id = slist[a]->id;
      slist.erase(slist.begin() + a); //Note: This calls the Sensor's destructor (I think)
      return id;
    }
  }
  cout << "SensorList::remove1: sensor not found" << endl;
}


//O(N) where N is number of sensors
///@return: ID previously used by the button
int SensorList::remove(int buttonid){
  int a;
  for (a = 0; a < slist.size(); a++){
    if (slist[a]->id == buttonid){
      slist.erase(slist.begin() + a); //Note: This calls the Sensor's destructor (I think)
      return buttonid;
    }
  }
  cout << "SensorList::remove2: sensor not found" << endl;
}


void SensorList::PrintALL(void){
  int a;
  cout << "Printing sensor pointers\n";
  for (a = 0; a < slist.size(); a++)
    cout << slist[a] << endl;
}

/*
//Look through all sensors in the list and add the location as needed
void SensorList::addLocation(wxSocketBase* sock, Location* loc){
  int a;
  for (a = 0; a < slist.size(); a++){
    if (slist[a]->sock == sock){
      slist[a].addLocation(loc);
      return;
    }
  }
}
*/

//Obtain the sensor to be updated
//NOTE: This returns the actual sensor's pointer, allowing it to be modified directly!
//FIXME: Make this function protected and add frame as a friend?
Sensor* SensorList::getSensor(wxSocketBase* sock){
  int a;
  for (a = 0; a < slist.size(); a++){
    if (slist[a]->sock == sock)
      return slist[a];
  }
  //cout << "Warning: Sensor not found in SensorList\n";
  return NULL;
}

//Obtain the sensor to be updated
//NOTE: This returns the actual sensor's pointer, allowing it to be modified directly!
//FIXME: Make this function protected and add frame as a friend?
Sensor* SensorList::getSensor(int buttonid){
  int a;
  for (a = 0; a < slist.size(); a++){
    if (slist[a]->id == buttonid)
      return slist[a];
  }
  cout << "Warning: Sensor not found in SensorList 2\n";
  return NULL;
}


vector<Location*> SensorList::getLocations(void){
  //Return location of each sensor
  vector<Location*> ret;
  ret.reserve(slist.size());
  int a;
  for (a = 0; a < slist.size(); a++){
    if (slist[a] != NULL){
      if (slist[a]->loc != NULL)
	ret.push_back(slist[a]->loc->copy());
      else
	cout << "Warning! SensorList::getLocations() skipping sensor with no location!" << endl;
    } else {
      cout << "Warning! SensorList has NULL sensor in it!" << endl;
    }
  }
  return ret;
}


