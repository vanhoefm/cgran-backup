/** aggsolvmgr.cpp: A wrapper class the frame uses to manage the aggregator and solver threads
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
#include "aggsolvmgr.h"

using namespace std;

/*-------AggSolvMgr implementation-----------*/

AggSolvMgr::AggSolvMgr(void){
  
  ///Creates buffer for passing items to aggregator and initializes aggregator thread
  aggbuf = new CondBuffer(AGGREGATOR_BUFFER_CAPACITY);
  aggthread = NULL; //Note: Thread does not run until the startThread() function is called.
  
  ///Creates buffer for passing items to solver and initializes solver thread
  solvbuf = new CondBuffer(SOLVER_BUFFER_CAPACITY);
  solvthread = NULL; //Note: Thread does not run until the startThread() function is called.

  ///Creates buffer for passing items to GUI frame
  framebuf = new CondBuffer(SOLVER_BUFFER_CAPACITY);
}


AggSolvMgr::~AggSolvMgr(void){
  stopThreads();
  delete aggbuf;
  delete solvbuf;
  delete framebuf;
}


void AggSolvMgr::add(Packet* pkt){
  aggbuf->addone(pkt);
}


void AggSolvMgr::signalDone(void){
  if (aggthread != NULL)
    aggthread->signalDone();
  else
    cout << "Warning! Cannot signal aggregator thread to be done when it isn't running!" << endl;
  
  if (solvthread != NULL)
    solvthread->signalDone();
  else
    cout << "Warning! Cannot signal solver thread to be done when it isn't running!" << endl;
}


//Start the thread.
//Note: This only should happen when all the sensors are connected.
//Note: It is legal (but inadvisable) to add items to the buffer BEFORE this occurs.
void AggSolvMgr::startThreads(ParamsWrapper* apwrap, ParamsWrapper* spwrap){
  //cout << "about to aggthread constructor" << endl;
  aggthread = new AggThread(aggbuf, solvbuf, apwrap);
  //cout << "about to create aggthread" << endl;
  wxThreadError err = aggthread->Create();
  if (err != wxTHREAD_NO_ERROR)
    throw "Unable to create aggregator thread\n";
  //cout << "about to run aggthread" << endl;
  err = aggthread->Run();
  if (err != wxTHREAD_NO_ERROR)
    throw "Unable to run aggregator thread\n";
  
  //cout << "about to solvthread constructor" << endl;
  solvthread = new SolverThread(solvbuf, framebuf, spwrap);
  //cout << "about to solvthread constructor" << endl;
  err = solvthread->Create();
  if (err != wxTHREAD_NO_ERROR)
    throw "Unable to create solver thread\n";
  //cout << "about to run solvthread" << endl;
  err = solvthread->Run();
  if (err != wxTHREAD_NO_ERROR)
    throw "Unable to run solver thread\n";
}


void AggSolvMgr::stopThreads(void){
  /// Destroys buffer and sending thread
  if (aggthread != NULL)
    aggthread->signalDone();
  if (solvthread != NULL)
    solvthread->signalDone();
  if (aggthread != NULL)
    aggthread->Wait();
  if (solvthread != NULL)
    solvthread->Wait();
  
  delete aggthread;
  delete solvthread;
  aggthread = NULL;
  solvthread = NULL;
}


vector<Measurement*> AggSolvMgr::grabSolverResults(void){
  vector<Measurement*> measvec;
  if (solvthread != NULL){
    deque<void*>* voidvec = framebuf->pollremove();
    if (voidvec != NULL){
      measvec.reserve(voidvec->size());
      //Copy measurements, converting them from void* in the process
      int a;
      for (a = 0; a < voidvec->size(); a++)
	measvec.push_back( (Measurement*) (*voidvec)[a] );
      delete voidvec;
    }
  }
  return measvec;
}


bool AggSolvMgr::isRunning(void){
  return (aggthread != NULL) && (solvthread != NULL);
}
