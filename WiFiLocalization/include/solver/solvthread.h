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
#ifndef SOLV_THREAD_H
#define SOLV_THREAD_H

#include <wx/wx.h>
#include <wx/thread.h>
#include "measurement.h"
#include "params.h"
#include "buffer.h"
#include "modeltemplates.h"

using namespace std;

//Time in ms between updates
#define SOLVER_QUEUE_DELAY 200


/**Solver thread
 * 
 * This thread performs localization on the measurements and passes them to the main thread.
 * 
 * @author Brian Shaw
 */
class SolverThread : public wxThread{
protected:
  bool done;
  CondBuffer* buffer;
  CondBuffer* framebuffer;
  LocalizationFramework* localizer;
public:
  /// Called from main thread, this notifies aggregator of changes to the 
  /// user's configuration preferences. Expect this to be infrequent.
  //int notifyConfigChange(AggParams* newparams);
  
  
  // ----- Built-in functions for thread declared here --- //
  SolverThread(CondBuffer* buf, CondBuffer* framebuf, ParamsWrapper* apwrap);
  // thread execution starts here
  virtual void *Entry();
  void signalDone(void);
};


///SolvParams: Container for Solver thread parameters
struct SolvParams : public Params{
  LocalizationFrameworkParams* fparam;
  
  void print(ostream* fs);
  void printcsv(ostream* fs);
  SolvParams* copy(void);
  ParamsType getType(void) { return PT_SOLV; };
};

#endif
