/** newintruderframe.h: Dialog window for creating a new intruder
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

#ifndef NRL_NEW_INTRUDER_FRAME_H
#define NRL_NEW_INTRUDER_FRAME_H


#include <wx/dialog.h>
#include <wx/spinctrl.h>
#include <wx/statline.h>

#include "simintruder.h"
#include "intruderpattern.h"
#include "app.h"

using namespace std;

class NewIntruderFrame: public wxDialog
{    
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
  
  /// Data members
  wxString mac;
  int x;
  int y;
  NRLPatternID pattern;
  Intruder** intruderptr; //Pointer to memory space owned by the main frame
  NRLApp* app;
  //For SquarePattern1
  Location patternvelocity;
  int numsteps;
  long totsteps;
  
public:
  NewIntruderFrame(); //Don't use this by itself
  NewIntruderFrame(wxWindow* parent, NRLApp* myapp, Intruder** newintruderptr);
  /// Creation
  bool Create(wxWindow* parent, NRLApp* myapp, Intruder** newintruderptr);
};

#endif
