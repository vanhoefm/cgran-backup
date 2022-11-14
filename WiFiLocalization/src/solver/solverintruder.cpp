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

#include "solverintruder.h"

using namespace std;

void TheIntruders::updateAll(std::deque<Measurement*>* update){
  int a;
  for (a = 0; a < intruders.size(); a++){
    bool deleteme = intruders[a]->update(update);
    if (deleteme){
      intruders.erase(intruders.begin() + a);
      a--;
    }
  }
}

bool Intruder::update(std::deque<Measurement*>* update){
  cout << "Updating Intruder with Measurement Deque\n";
  if (update != NULL){
    //Find which measurement pertains to us.
    //If none do, increment our "coasts" and check for deletion
    int a;
    for (a = 0; a < update->size(); a++){
      //Check to see if it's our update
      if ((*update)[a]->sameMac(mac)){
	//It's ours! Update our position accordingly
	wascoasted = 0;
	Location oldloc = lcn;
	lcn = (*update)[a]->getPosition();
	velocity = lcn - oldloc;
	uncertainty = (*update)[a]->getUncertainty();//FIXME: improve
	//FIXME: Retrieve other info if present
	//Get rid of measurement
	update->erase(update->begin()+a);
	return false;
      }
    }
  }     
  //No matches were found, or no measurements provided. 
  //This intruder may no longer exist, or it could be a momentary glitch.
  //Update coast information
  wascoasted++;
  if (wascoasted > MAX_INTRUDER_COASTS)
    return true;
  else
    return false;
}


Intruder::Intruder(Measurement* meas, NRLApp* app) : 
	  NRLIntruder(meas->getMac(), meas->getPositionHeap(), app){
  time_first_appeared = meas->getDetectionTime();
  velocity = Location(0, 0, 0); //The location and uncertainty should get Kalman smoothed later on
  uncertainty = meas->getUncertainty();
  delete meas;
  //button = new wxBitmapButton(app->panel, wxID_OK, *global_pic_e,
 //			      computeItemPosition(&lcn, 175, 175), wxSize(175, 175), 0);
}


Intruder::~Intruder(void){
  //delete time_first_appeared;
  //delete button;
}


TheIntruders::TheIntruders(NRLApp* app) : NRLTheIntruders(app){
  
}