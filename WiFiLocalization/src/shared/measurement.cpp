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

//FIXME: Split the "Measurement" class into separate 
//PktGroup (for aggregated packet triplets) and Measurement (for solver results)

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
  
  //Fill in data fields with available information
  position = NULL;
  uncertainty = NULL;
  if (pktarray->size() > 0){
    mac = (*pktvec)[0]->mac->copy();
    timedetected = (*pktvec)[0]->toa->copy(); //Oldest packet is always at the front of the array, due to how the aggregator works
  } else {
    mac = NULL;
    timedetected = NULL;
  }
  
  return sanityCheck();
}


//Make sure a measurement is sane
bool Measurement::sanityCheck(void){
  //FIXME
  return true;
}


Measurement::~Measurement(void){
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
  
  if (pktarray != NULL)
    *fs << "(contains cluster)" << endl;
  else
    *fs << "(does not contain cluster)" << endl;
  
  if (mac != NULL)
    mac->print(fs);
  if (timedetected != NULL){
    *fs << "First time detected:\n";
    timedetected->print(fs);
  }
  if (position != NULL){
    *fs << "Calculated Location:\n";
    position->print(fs);
  }
  if (uncertainty != NULL){
    *fs << "Uncertainty:\n";
    uncertainty->print(fs);
  }
  
  *fs << "-------------------\n";
}

void Measurement::printcsv(ostream* fs){
  if (mac == NULL)
    return;
  //Must have this data to print to CSV.
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
  if (mac != NULL)
    meas->mac = mac->copy();
  else
    meas->mac = NULL;
  if (position != NULL)
    meas->position = position->copy();
  else
    meas->position = NULL;
  if (timedetected != NULL)
    meas->timedetected = timedetected->copy();
  else
    meas->timedetected = NULL;
  if (uncertainty != NULL)
    meas->uncertainty = uncertainty->copy();
  else
    meas->uncertainty = NULL;
  
  //Copy pktarray
  if (pktarray != NULL){
    vector<Packet*>* pktarray2 = new vector<Packet*>;
    pktarray2->reserve(pktarray->size());
    int a;
    for (a = 0; a < pktarray->size(); a++)
      pktarray2->push_back( (*pktarray)[a]->copy() );
    meas->pktarray = pktarray2;
  }
  
  return meas;
}


vector<Location*> Measurement::getLocations(void){
  vector<Location*> locvec;
  int a;
  for (a = 0; a < pktarray->size(); a++)
    locvec.push_back( (*pktarray)[a]->getloc() );
  return locvec;
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
  ///This function returns a pointer to the measurement's own MAC
  ///It is intended to be obtained immediately before the measurement is deleted
  return mac;
}

bool Measurement::sameMac(MacAddr* intrudermac){
  return mac->sameMac(intrudermac);
}

