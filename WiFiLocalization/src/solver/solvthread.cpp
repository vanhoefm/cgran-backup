/** solvthread.h: Solver thread that performs localization
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
#include "solvthread.h"
#include <vector>
#include <deque>
#include "modeltemplates.h"
#include "frameworks.h"

using namespace std;


/*---------SolvParams implementation-------*/

void SolvParams::print(ostream* fs){
  *fs << "/-----Solver parameters-----\\\n";
  fparam->print(fs);
  *fs << "\\-------------------------------/\n";
}

void SolvParams::printcsv(ostream* fs){
  fparam->printcsv(fs);
}

SolvParams* SolvParams::copy(void){
  SolvParams* mycopy = new SolvParams();
  mycopy->fparam = fparam->copy();
  return mycopy;
}


/*---------SolverThread implementation---------*/

/// Thread constructor used to pass in and store arguments 
// (only one is needed)
SolverThread::SolverThread(CondBuffer* buf, CondBuffer* framebuf, ParamsWrapper* apwrap) : wxThread(wxTHREAD_JOINABLE){
  buffer = buf;
  framebuffer = framebuf;
  done = false;
  //Create localization framework (algorithm for localization)
  if (apwrap == NULL)
    throw "Solver needs non-NULL parameter container!\n";
  Params* param = apwrap->getParams();
  if (param->getType() != PT_SOLV)
    throw "Solver received incorrect type of parameters!\n";
  //Unpack aggregator parameters
  SolvParams* param2 = (SolvParams*) param;
  
  //cout << "About to create aggregator object according to its type" << endl;
  //cout << "param2 is " << param2 << endl;
  //cout << "param2->fparam is " << param2->fparam << endl;
  param2->fparam->getType();
  //cout << "Gettype worked" << endl;
  
  switch (param2->fparam->getType()){
    case PT_5STEP:
      //cout << "param2->fparam didn't segfault" << endl;
      localizer = new FiveStepFramework( (FiveStepFrameworkParams*) param2->fparam);
      cout << "new framework created OK" << endl;
      break;
    //Other legal frameworks go here
    default:
      throw "Solver's localization framework parameters are the wrong type!\n";
  }   
}


/// Thread execution function
void* SolverThread::Entry(void){
  int a;
  cout << "INFO: solver thread running" << endl;
  std::deque<void*>* items;
  std::vector<Measurement*> measurements;
  std::vector<void*> voidvec;
  
  while (!done){
    //Grab everything in the buffer
    //cout << "About to grab from buffer (solver thread)" << endl;
    items = buffer->remove(SOLVER_QUEUE_DELAY, This(), true); //Blocks when buffer is empty
    
    try {
      if (items != NULL){
	if (items->size() > 0){
	  //cout << "About to convert items to measurements" << endl;
	  //Copy from recv'd deque to a Measurement vector
	  for (a = 0; a < items->size(); a++)
	    measurements.push_back( (Measurement*) (*items)[a] );
	  cout << "About to localize each measurement" << endl;
	  //Process measurements in place using Models provided
	  for (a = 0; a < measurements.size(); a++)
	    localizer->localize(measurements[a]);
	  //cout << "About to convert measurements to items" << endl;
	  //Copy from recv'd deque to a Measurement vector
	  for (a = 0; a < items->size(); a++)
	    voidvec.push_back( (void*) measurements[a] );
	  //cout << "About to add items to frame buffer" << endl;
	  framebuffer->add(voidvec);
	  measurements.clear();
	  voidvec.clear();
	  //cout << "Successfully added items to frame buffer" << endl;
	}
      }
    } catch (const char* err){
      cout << "Solver thread error: " << err << endl;
    } catch (...){
      cout << "Other error in solver thread" << endl;
    }
    
    done = done || TestDestroy(); //REQUIRED: Check to see if the thread needs to exit   
  }
  
  //Exit and join with main thread
  delete items;
  return NULL;
}


void SolverThread::signalDone(void){
  done = true;
}


