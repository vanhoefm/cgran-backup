/** setcalibrationdialog.h: Dialog window for labelling an intruder for calibration
 * 
 * This implementation is based on the Personal Record example
 * @author Brian Shaw
 * @author Julian Smart
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
/////////////////////////////////////////////////////////////////////////////
// Name:        personalrecord.h
// Purpose:     Dialog to get name, age, sex, and voting preference
// Author:      Julian Smart
// Created:     02/28/04 06:52:49
// Copyright:   (c) 2004, Julian Smart
// Licence:     wxWindows license
/////////////////////////////////////////////////////////////////////////////

#ifndef NRL_CALIBRATION_DIALOG_H
#define NRL_CALIBRATION_DIALOG_H


#include <wx/dialog.h>
#include <wx/statline.h>
#include <wx/spinctrl.h>

#include "macaddr.h"
#include "location.h"
#include "app.h"

using namespace std;

class CalibDialog: public wxDialog{    
  DECLARE_EVENT_TABLE()
private:
  /// Set or reset all fields to their default values
  void Init();
  
  /// Creates the controls and sizers
  void CreateControls();
  /// Sets the validators for the dialog controls
  void SetDialogValidators();
  /// Sets the help text for the dialog controls
  void SetDialogHelp();
  
  //// NewIntruderFrame event handler declarations
  /// wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_RESET
  void OnResetClick( wxCommandEvent& event );
  /// wxEVT_COMMAND_BUTTON_CLICKED event handler for wxID_HELP
  void OnHelpClick( wxCommandEvent& event );
  /// Returns the data from the dialog
  void OnOKClick( wxCommandEvent& event );
  /// Quits the dialog
  void OnCancelClick( wxCommandEvent& event );
  
  /// Data members
  wxString mac;
  int32_t x;
  int32_t y;
  MacAddr** macptr; //Pointer to memory space owned by the main frame
  Location** locptr;
  NRLApp* app;
  CalibDialog** calibdialog; //Pointer to main frame's pointer to us. Used for determining if the dialog box is open.
  
public:
  /// Default constructor and destructor. Use Create() to initialize everything.
  bool Create(wxWindow* parent, NRLApp* myapp, CalibDialog** dialog, MacAddr** macptrd, Location** locptrd);
};

#endif
