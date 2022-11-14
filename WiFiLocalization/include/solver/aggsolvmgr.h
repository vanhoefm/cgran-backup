/** aggsolvmgr.h: A wrapper class the frame uses to manage the aggregator and solver threads
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
#ifndef AGG_SOLV_MGR_H
#define AGG_SOLV_MGR_H

#include "aggthread.h"
#include "solvthread.h"
#include <vector>

using namespace std;

///Wrapper for both aggregator and solver threads
class AggSolvMgr{
  AggThread* aggthread;
  CondBuffer* aggbuf;
  SolverThread* solvthread;
  CondBuffer* solvbuf;
  CondBuffer* framebuf;
public:
  AggSolvMgr(void);
  ~AggSolvMgr(void);
  void signalDone(void);
  
  void startThreads(ParamsWrapper* apwrap, ParamsWrapper* spwrap);
  void stopThreads(void);
  bool isRunning(void);
  void add(Packet* pkt);
  vector<Measurement*> grabSolverResults(void);
};

#endif
