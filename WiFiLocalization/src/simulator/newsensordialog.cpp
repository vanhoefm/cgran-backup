/** newsensordialog.cpp: Dialog window for creating a new sensor
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
// Name:        personalrecord.cpp
// Purpose:     Dialog to get name, age, sex, and voting preference
// Author:      Julian Smart
// Created:     02/28/04 06:52:49
// Copyright:   (c) 2004, Julian Smart
// Licence:     wxWindows license
/////////////////////////////////////////////////////////////////////////////

#include "newsensordialog.h"
#include "gui_sizes.h"
#include "globalparams.h"
#include <wx/wx.h>
#include <wx/valtext.h>
#include <wx/valgen.h>

using namespace std;

/*!
 * Control identifiers
 */

enum {
  NRLID_X = 10002,
  NRLID_Y = 10003,
  ID_RESET = 10004
};

/*!
 * NewSensorDialog event table definition
 */

BEGIN_EVENT_TABLE( NewSensorDialog, wxDialog )
  EVT_BUTTON( ID_RESET, NewSensorDialog::OnResetClick )
  EVT_BUTTON( wxID_HELP, NewSensorDialog::OnHelpClick )
  EVT_BUTTON( wxID_OK, NewSensorDialog::OnOKClick )
  EVT_BUTTON( wxID_CANCEL, NewSensorDialog::OnCancelClick )
END_EVENT_TABLE()

//BUG: Clicking on Help while this is open doesn't show whether the dialog or the main help gets displayed. This may not matter, though.
 
/// Initialisation
void NewSensorDialog::Init(void){
  x = 0;
  y = 0;
}


//Call this function immediately after the constructor
//If it returns a negative value, delete the object
bool NewSensorDialog::Create(wxWindow* parent, NRLApp* theapp, NewSensorDialog** dialog, Location** newsensorptr){
  //Initialize the dialog base class
  SetExtraStyle(wxDIALOG_EX_CONTEXTHELP);
  if (!wxDialog::Create(parent, -1, _("New Sensor Information")))
    return false;
  
  app = theapp;
  sensorptr = newsensorptr;
  newsensordialog = dialog;
  Init();
  
  CreateControls();
  SetDialogHelp();
  SetDialogValidators();
  
  // Fit the dialog to the minimum size dictated by the sizers
  GetSizer()->Fit(this);
  
  // Ensure that the dialog cannot be sized smaller than the minimum size
  GetSizer()->SetSizeHints(this);

  // Center the dialog
  Center();
  return true;
}


/*!
 * wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_RESET
 */
void NewSensorDialog::OnResetClick(wxCommandEvent& event){
  Init();
  TransferDataToWindow();
}

/*!
 * wxEVT_COMMAND_BUTTON_CLICKED event handler for wxID_HELP
 */
//FIXME: Use something more computationally efficient than a message box
void NewSensorDialog::OnHelpClick( wxCommandEvent& event ){
  
  wxString helpText =
    wxT("Please enter a location for the new sensor in centimeters.\n")
    wxT("X is the horizontal position, Y is the vertical position\n\n");

  wxMessageBox(helpText,
    wxT("New Sensor Dialog Help"),
    wxOK|wxICON_INFORMATION, this);
}


///OK button clicked (event handler for wxID_OK)
void NewSensorDialog::OnOKClick( wxCommandEvent& event ){
  bool datavalid = true;
  //Transfer data from the dialog window to our variables
  TransferDataFromWindow();

  if (*sensorptr != NULL){
    //Timer hasn't had a chance to save the last new sensor we made! Ask user to wait 2 * update period
    cout << "Please wait " << (((float) UPDATE_RATE) / 500) << " seconds before creating a new sensor." << endl;
    return;
  }
  
  //Sanitize sensor data or return //FIXME: This just prints it
  //cout << "X: " << x << " cm" << endl;
  //cout << "Y: " << y << " cm" << endl;
  
  if (!datavalid){
    cout << "Sensor data entered was invalid!" << endl;
    Init();
    return; //without closing dialog!
  }
  
  *sensorptr = new Location(cm_to_meters(x), cm_to_meters(y), 0); //Frame's SensorList class will create the actual sensor for us
  *newsensordialog = NULL;
  //cout << "OK was clicked and data passed back (sensor)" << endl;
  Close();
  Destroy();
}


void NewSensorDialog::OnCancelClick( wxCommandEvent& event ){
  //NOTE: This gets called even when OK closes the dialog!
  *newsensordialog = NULL;
  Close();
  Destroy();
}


/*!
 * Control creation for NewSensorDialog
 */

void NewSensorDialog::CreateControls(){
  //BUG: The spin controls only handle integers
  //NOTE: Set the max/min values for an on-screen location in location.h
  
  // A top-level sizer
  wxBoxSizer* topSizer = new wxBoxSizer(wxVERTICAL);
  this->SetSizer(topSizer);
  // A second box sizer to give more space around the controls
  wxBoxSizer* boxSizer = new wxBoxSizer(wxVERTICAL);
  topSizer->Add(boxSizer, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5);
  // A friendly message
  wxStaticText* descr = new wxStaticText( this, wxID_STATIC,
      _("Please select the sensor's location."),
      wxDefaultPosition, wxDefaultSize, 0 );
  boxSizer->Add(descr, 0, wxALIGN_LEFT|wxALL, 5);
  // Spacer
  boxSizer->Add(5, 5, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5);
  
  // A horizontal box sizer to contain location's X and Y
  wxBoxSizer* locbox = new wxBoxSizer(wxHORIZONTAL);
  boxSizer->Add(locbox, 0, wxGROW|wxALL, 5);
  // Label for the X control
  wxStaticText* xLabel = new wxStaticText ( this, wxID_STATIC,
      wxT("&X:"), wxDefaultPosition, wxDefaultSize, 0 );
  locbox->Add(xLabel, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);
  // A spin control for the X coordinate
  wxSpinCtrl* xSpin = new wxSpinCtrl ( this, NRLID_X,
      wxEmptyString, wxDefaultPosition, wxSize(60, -1),
      wxSP_ARROW_KEYS, MIN_X_POS, MAX_X_POS, 25 );
  locbox->Add(xSpin, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);
  // Label for the Y control
  wxStaticText* yLabel = new wxStaticText ( this, wxID_STATIC,
      wxT("&Y:"), wxDefaultPosition, wxDefaultSize, 0 );
  locbox->Add(yLabel, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);
  // A spin control for the Y coordinate
  wxSpinCtrl* ySpin = new wxSpinCtrl ( this, NRLID_Y,
      wxEmptyString, wxDefaultPosition, wxSize(60, -1),
      wxSP_ARROW_KEYS, MIN_Y_POS, MAX_Y_POS, 25 );
  locbox->Add(ySpin, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);
  
  
  // A dividing line before the OK and Cancel buttons
  wxStaticLine* line = new wxStaticLine ( this, wxID_STATIC,
      wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
  boxSizer->Add(line, 0, wxGROW|wxALL, 5);
  // A horizontal box sizer to contain Reset, OK, Cancel and Help
  wxBoxSizer* okCancelBox = new wxBoxSizer(wxHORIZONTAL);
  boxSizer->Add(okCancelBox, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5);
  // The Reset button
  wxButton* reset = new wxButton( this, ID_RESET, wxT("&Reset"),
      wxDefaultPosition, wxDefaultSize, 0 );
  okCancelBox->Add(reset, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);
  // The OK button
  wxButton* ok = new wxButton ( this, wxID_OK, wxT("&OK"),
      wxDefaultPosition, wxDefaultSize, 0 );
  okCancelBox->Add(ok, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);
  // The Cancel button
  wxButton* cancel = new wxButton ( this, wxID_CANCEL,
      wxT("&Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
  okCancelBox->Add(cancel, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);
  // The Help button
  wxButton* help = new wxButton( this, wxID_HELP, wxT("&Help"),
      wxDefaultPosition, wxDefaultSize, 0 );
  okCancelBox->Add(help, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);
}

// Set the validators for the dialog controls
void NewSensorDialog::SetDialogValidators(){
  FindWindow(NRLID_X)->SetValidator(
      wxGenericValidator(& x));
  FindWindow(NRLID_Y)->SetValidator(
      wxGenericValidator(& y));
}


// Sets the help text for the dialog controls
void NewSensorDialog::SetDialogHelp(void){
  wxString xHelp = wxT("Enter sensor's X coordinate (in cm)"); 
  wxString yHelp = wxT("Enter sensor's Y coordinate (in cm)");
  FindWindow(NRLID_X)->SetHelpText(xHelp);
  FindWindow(NRLID_X)->SetToolTip(xHelp);
  FindWindow(NRLID_Y)->SetHelpText(yHelp);
  FindWindow(NRLID_Y)->SetToolTip(yHelp);
}
