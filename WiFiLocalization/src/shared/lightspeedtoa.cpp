/** lightspeedtoa.cpp: Simple time-based distance calculator
 * uses customizable speed of light (free space, in air, through solid material, etc)
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
#include "lightspeedtoa.h"
#include <iostream>

using namespace std;

/* -------- LightspeedToAParams implementation -------- */

LightspeedToAParams* LightspeedToAParams::copy(void){
  LightspeedToAParams* p = new LightspeedToAParams();
  p->c_nsec = c_nsec; //double is easily copied
  p->timecorr = timecorr->copy();
  return p;
}


void LightspeedToAParams::printcsv(ostream* fs){
  *fs << c_nsec;
  *fs << ";";
  timecorr->printcsv(fs);
}


void LightspeedToAParams::print(ostream* fs){
  *fs << "/---Lightspeed Model Parameters---\\" << endl;
  *fs << "Speed of Light: " << c_nsec << endl;
  *fs << "RF Hardware Delay: ";
  timecorr->print(fs);
  *fs << "\\---------------------------------/" << endl;
}

/* -------- LightspeedToA implementation --------- */

bool LightspeedToA::setparams(ToADistanceCalculatorParams* params){
  if (params->getType() != PT_LIGHTSPEED){
    cout << "Warning! Unable to set Lightspeed ToA parameters (wrong parameter type)" << endl;
    return false;
  }
  //Clear out correction factor vector
  delete timecorr;
  /*
  int a;
  for (a = 0; a < corrloc.size(); a++){
    delete corrloc[a];
    delete corrtime[a];
  }
  */
  
  LightspeedToAParams* p = (LightspeedToAParams*) params;
  c_nsec = p->c_nsec;
  timecorr = p->timecorr;
  //corrloc = p->corrloc;
  //corrtime = p->corrtime;
  delete params;
  return true;
}


//O(n * s)
vector<double> LightspeedToA::toa_to_dist(Measurement* meas){
  vector<double> retdist;
  retdist.reserve(meas->pktarray->size());
  int a;
  for (a = 0; a < meas->pktarray->size(); a++){
    Packet* pkt = (*(meas->pktarray))[a];
    double dist;
    if (pkt->isCalibrationPkt()){
      //Calibration packet: set time offset appropriately
      cout << "ToA Using packet for calibration" << endl;
      //addTimeCorrection(pkt);
      dist = pkt->getcalibdist();
    } else {
      //Typical case
      Timestamp* ts = correctTime(pkt);
      //cout << "LightspeedToA::toa_to_dist about to call ts->todouble" << endl;
      double timediff = ts->todouble();
      delete ts;
      dist = timediff * c_nsec;
    }
    retdist.push_back(dist);
  }
  //cout << "About to return from toa_to_dist" << endl;
  return retdist;
}

  
Timestamp* LightspeedToA::dist_to_toa(double dist, Timestamp* time){
  Timestamp* ts = new Timestamp(time->gettime() + timecorr->gettime(),
				time->getnsec() + timecorr->getnsec() + dist / c_nsec);
  ts->normalize();
  return ts;
}


//Helper function for time correction
Timestamp* LightspeedToA::correctTime(Packet* pkt){
  if (pkt == NULL)
    throw "Null packet in LightspeedToA::correctTime";
  Location loc = pkt->getloc2();
  Timestamp* time = pkt->timediff();
  int a;
  return *time - *timecorr;
  /*
  for (a = 0; a < corrloc.size(); a++){
    if ( corrloc[a]->isSame(&loc) ){
      //Matched, so apply the time correction
      Timestamp* ret = *time - corrtime[a];
      delete time;
      return ret;
    }
  }
  
  //If we got here, that means no correction factor was present for this sensor
  cout << "Warning! No correction factor available for sensor at ";
  loc.print(&cout);
  return time; //No correction applied
  */
}


/*
void LightspeedToA::addTimeCorrection(Packet* pkt){
  //NOTE: Adds time correction the first time that packet is calibrated, only.
  //BUG: in the future, this should be Kalman smoothed.
  int a;
  for (a = 0; a < corrloc.size(); a++){
    Location lcn = pkt->getloc2();
    if ( corrloc[a]->isSame(&lcn) )
      return; //Already have this value, no need to continue
  }
  
  //If we got here, we don't already have the time correction for this packet.
  corrloc.push_back(pkt->getloc());
  corrtime.push_back(dist_to_toa(pkt->getcalibdist()));
}
*/


LightspeedToA::LightspeedToA(LightspeedToAParams* params){
  timecorr = NULL; //So delete doesn't fail
  //cout << "Creating LightspeedToA" << endl;
  setparams(params);
}


LightspeedToA::~LightspeedToA(void){
  /*
  int a;
  for (a = 0; a < corrloc.size(); a++){
    delete corrloc[a];
    delete corrtime[a];
  }
  */
  delete timecorr;
}

  
