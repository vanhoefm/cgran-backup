/** Intruder: An unwanted visitor node transmitting packets
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

#include "simintruder.h"

using namespace std;


std::vector<int> TheIntruders::updateAll(void){
  int a;
  std::vector<int> thosedeleted;
  for (a = 0; a < intruders.size(); a++){
    bool deleteme = intruders[a]->update();
    if (deleteme){
      cout << "Deleting intruder " << intruders[a]->buttonid << endl;
      thosedeleted.push_back(intruders[a]->buttonid);
      delete intruders[a]->button;
      intruders.erase(intruders.begin() + a);
    }
  }
  return thosedeleted;
}


bool Intruder::update(void){
  cout << "Updating intruder " << buttonid << endl;
  //Calculate how long (in wall time) we have been alive
  Timestamp currtime = Timestamp(); //autofills using GetTimeOfDay()
  Timestamp* timealive = currtime - time_first_appeared;
  //Get data from our pattern
  Location* newvel = mypattern->getNewVelocity(timealive);
  //BEGIN DEBUG
  if (newvel != NULL)
    newvel->print(&cout);
  //END DEBUG
  bool intruderDead = mypattern->isIntruderDead(timealive);
  delete timealive;
    
  if (newvel != NULL){
    cout << "Updating Intruder with New Velocity (a Location)\n";
    //delete velocity;
    velocity = *newvel;
    cout << 5.1 << endl;
    delete newvel;
  }
  //Update our location
  lcn = lcn + velocity;
  //Update the image we are using as well as where our button is
  updateIcon();
  
  return intruderDead; //Return value indicates whether or not this should be deleted
}


//Intruder::Intruder(Location* loc, Location* vel, NRLApp* app) : NRLIntruder(loc, vel, app){
Intruder::Intruder(MacAddr* mac, Location* loc, IntruderPattern* pattern, NRLApp* app) : NRLIntruder(mac, loc, app){
  time_first_appeared = new Timestamp();
  mypattern = pattern;
}


Intruder::~Intruder(void){
  //delete time_first_appeared;
}


TheIntruders::TheIntruders(NRLApp* app) : NRLTheIntruders(app){
  
}


///Add a new intruder to our list
void TheIntruders::add(Intruder* intruder){
  if (intruder != NULL){
    cout << "Adding intruder to list" << endl;
    intruders.push_back(intruder);
  }
}


///Add a vector of intruders to our list
void TheIntruders::add(std::vector<Intruder*> intruderlist){
  int a;
  for (a = 0; a < intruderlist.size(); a++)
    add(intruderlist[a]);
}

