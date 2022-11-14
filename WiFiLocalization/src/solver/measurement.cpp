/** Measurement.cpp: Implementation of Measurement class
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

#include "measurement.h"

using namespace std;

Measurement::Measurement(){
  //These are needed so the measurement can be safely deleted if invalid.
  pktarray = NULL;
  mac = NULL;
  position = NULL;
  uncertainty = NULL;
  timedetected = NULL;
}

bool Measurement::measbuilder(std::vector<Packet*>* pktvec){
  if (pktvec == NULL){
    cout << "NULL packet vector being used to build measurement" << endl;
    return false;
  }
  if (pktvec->size() < 3){ //FIXME: MIN_REQUIRED_PKTS
    cout << "Not enough packets in the packet vector to build a measurement" << endl;
    return false;
  }
  //NOTE: We need to copy all of the packets into new memory because
  //the aggregator may delete them all
  pktarray = new std::vector<Packet*>();
  int a;
  for (a = 0; a < pktvec->size(); a++)
    pktarray->push_back((*pktvec)[a]->copy());
  
  //cout << "Creating data for measurement" << endl;
  position = new Location(0, 0, 0); //FIXME: This exists ONLY so that print won't fail.
  uncertainty = new Location(0, 0, 0); //FIXME: This exists ONLY so that print won't fail.
  mac = (*pktvec)[0]->mac->copy();
  timedetected = (*pktvec)[0]->toa->copy(); //Oldest packet is always at the front of the array, due to how the aggregator works
  
  //Check if everything is valid. FIXME 
  return true;
}

Measurement::~Measurement(void){
  //cout << "Deleting a measurement\n"
  int a;
  if (pktarray != NULL){
    for (a = 0; a < pktarray->size(); a++)
      delete (*pktarray)[a];
  }
  delete pktarray;
  delete mac;
  delete position;
  delete uncertainty;
  delete timedetected;
}

void Measurement::print(ostream* fs){
  *fs << "----Measurement----\n";
  mac->print(fs);
  *fs << "First time detected:\n";
  timedetected->print(fs);
  *fs << "Calculated Location:\n";
  position->print(fs);
  *fs << "Uncertainty:\n";
  uncertainty->print(fs);
  *fs << "-------------------\n";
}

void Measurement::printcsv(ostream* fs){
  mac->printcsv(fs);
  *fs << ";";
  timedetected->printcsv(fs);
  *fs << ";";
  position->printcsv(fs);
  *fs << ";";
  uncertainty->printcsv(fs);
}

Measurement* Measurement::copy(void){
  Measurement* meas = new Measurement();
  meas->mac = mac->copy();
  meas->position = position->copy();
  meas->timedetected = timedetected->copy();
  meas->uncertainty = uncertainty->copy();
}


Location Measurement::getPosition(void){
  return Location(*position);
}

Location* Measurement::getPositionHeap(void){
  return new Location(*position);
}

Location Measurement::getUncertainty(void){
  return Location(*uncertainty);
}

Timestamp* Measurement::getDetectionTime(void){
  return new Timestamp(timedetected);
}

MacAddr* Measurement::getMac(void){
  ///NOTE: This function returns a pointer to the measurement's own MAC
  ///It is intended to be obtained immediately before the measurement is deleted
  return mac;
}

bool Measurement::sameMac(MacAddr* intrudermac){
  return mac->sameMac(intrudermac);
}

