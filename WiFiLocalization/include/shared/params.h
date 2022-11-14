/** params.h: Parameter container definitions
 * (all key parameters are kept in containers for easy access and on-the-fly modification)
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
#ifndef PARAMS_H
#define PARAMS_H

using namespace std;

#include <wx/thread.h>

#include "templates.h"
#include "timestamp.h"
#include "paramstypes.h" //NOTE: Any time a new Params is created, a Type must be added here


///Params: Abstract wrapper for all parameter structs
struct Params : public Printable{
  virtual void print(ostream* fs) = 0;
  virtual void printcsv(ostream* fs) = 0;
  virtual Params* copy(void) = 0;
  //Functions to see what type of parameter struct this is
  virtual ParamsType getType(void) = 0;
};


///ParamsWrapper: Wrapper class for all parameter containers
///Allows safe access to parameters that can be modified at any time
//FUTURE: Modify this class to use templates (ParamsWrapper<AggParams> for ex.)
class ParamsWrapper : public Printable{
protected:
  Params* params;
  wxMutex lock;

public:
  ParamsWrapper(Params* defaultparams);
  ~ParamsWrapper(void);
  
  ParamsWrapper* copy(void);
  void print(ostream* fs);
  void printcsv(ostream* fs);
  Params* getParams(void);
  void setParams(Params* newparams);
  
  ParamsType paramsType(void);
};


//extern ParamsWrapper* aggparams_w;

#endif
