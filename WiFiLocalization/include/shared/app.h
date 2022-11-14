/* app.h: Header for NRLMQP GUI app
 * Note: This header is used for two separate implementations of the NRLApp
 * (one for the solver, one for the simulator)
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

#ifndef NRL_SOLVER_APP_H
#define NRL_SOLVER_APP_H

//Include wxWidgets
#include "wx/wx.h"

//Application class
class NRLApp : public wxApp{
private:
  //Called on application startup
  virtual bool OnInit();
  virtual int OnExit();
  
public:
  wxPanel* panel; //Used by the Frame when making Sensors and Intruders visible
};

#endif
