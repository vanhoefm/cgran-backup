/** solverframe.h: Application frame class for solver GUI
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

#ifndef NRL_FRAME_H
#define NRL_FRAME_H

#include <wx/wx.h>
#include <wx/socket.h>

#include "logthread.h"
#include "aggsolvmgr.h"
#include "sensor.h"
#include "app.h"
#include "solverintruder.h"
#include "setcalibrationdialog.h"
#include "macaddr.h"
#include "location.h"

using namespace std;

#define NUM_SENSORS_DESIRED	3

class NRLFrame : public wxFrame{
public:
  //Constructor
  NRLFrame(const wxString& title, NRLApp* app);
  
  //Event handlers
  void OnQuit(wxCommandEvent& event);
  void OnAbout(wxCommandEvent& event);
  void OnLogData(wxCommandEvent& event);
  void OnItemClick(wxCommandEvent& event);
  void OnServerStart(wxCommandEvent& WXUNUSED(event));
  void OnServerEvent(wxSocketEvent& WXUNUSED(event));
  void OnSocketEvent(wxSocketEvent& event);
  void OnCalib(wxCommandEvent& event);
  
private:
  NRLApp* theapp;
  SensorList sensorlist;
  TheIntruders* intruders;
  void updateIntruders(std::deque<Measurement*>* update);
  void makeSensorVisible(Sensor* sens);
  void makeSensorInvisible(Sensor* sens);
  
  TheLoggerThread* logger;
  AggSolvMgr* aggsolvmgr;
  //Temporary solver showing stuff
  
  //Parameter wrappers: FIXME put these in aggsolvmgr?
  ParamsWrapper* aggparams;
  ParamsWrapper* solverparams;
  
  //Event handling
  DECLARE_EVENT_TABLE()
  
  //Socket handling
  wxSocketServer *m_server;
  bool m_server_ok;
  std::vector<wxSocketBase*> connections;
  
  //Calibration
  CalibDialog* calibdialog;
  Location* calibloc;
  MacAddr* calibmac; //BUG: Only one of these can exist at a time
  
  //Update timer
  wxTimer* m_timer;
  void OnTimerEvent(wxTimerEvent& event);
  
  //Lots of button events, one for each button in the button pool
  std::vector<bool> buttonsavailable;
  int getNextButtonID(void);
  void freeButtonID(int id);
  
  void onButtonClick(int buttonid);
  void onButtonClick0(wxCommandEvent& event);
  void onButtonClick1(wxCommandEvent& event);
  void onButtonClick2(wxCommandEvent& event);
  void onButtonClick3(wxCommandEvent& event);
  void onButtonClick4(wxCommandEvent& event);
  void onButtonClick5(wxCommandEvent& event);
  void onButtonClick6(wxCommandEvent& event);
  void onButtonClick7(wxCommandEvent& event);
  void onButtonClick8(wxCommandEvent& event);
  void onButtonClick9(wxCommandEvent& event);
  void onButtonClick10(wxCommandEvent& event);
};

#endif
