/** setcalibrationdialog.cpp: Dialog window for designating a MAC address as calibration
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

#include "setcalibrationdialog.h"
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
  NRLID_MAC = 10001,
  NRLID_X,
  NRLID_Y,
  ID_RESET = 10004
};

/*!
 * CalibDialog event table definition
 */

BEGIN_EVENT_TABLE( CalibDialog, wxDialog )
  EVT_BUTTON( ID_RESET, CalibDialog::OnResetClick )
  EVT_BUTTON( wxID_HELP, CalibDialog::OnHelpClick )
  EVT_BUTTON( wxID_CANCEL, CalibDialog::OnCancelClick )
  EVT_BUTTON( wxID_OK, CalibDialog::OnOKClick )
END_EVENT_TABLE()

//BUG: Clicking on Help while this is open doesn't show whether the dialog or the main help gets displayed. This may not matter, though.
 
/// Initialisation
void CalibDialog::Init(void){
  mac = _("aa:bb:cc:dd:ee:ff"); //default Mac address
}


//Call this function immediately after the constructor
//If it returns a negative value, delete the object
bool CalibDialog::Create(wxWindow* parent, NRLApp* myapp, CalibDialog** dialog, MacAddr** macptrd, Location** locptrd){
  //Initialize the dialog base class
  SetExtraStyle(wxDIALOG_EX_CONTEXTHELP);
  if (!wxDialog::Create(parent, -1, _("Designate Calibration Intruder")))
    return false;
  
  app = myapp;
  macptr = macptrd;
  locptr = locptrd;
  calibdialog = dialog;
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
void CalibDialog::OnResetClick(wxCommandEvent& event){
  Init();
  TransferDataToWindow();
}

/*!
 * wxEVT_COMMAND_BUTTON_CLICKED event handler for wxID_HELP
 */
//FIXME: Use something more computationally efficient than a message box
void CalibDialog::OnHelpClick( wxCommandEvent& event ){
  
  wxString helpText =
    wxT("Please enter some information about the calibrating intruder.\n")
    wxT("X and Y are the position of the intruder, in centimeters\n\n"); //FIXME
    
  wxMessageBox(helpText,
    wxT("Set Calibration Dialog Help"),
    wxOK|wxICON_INFORMATION, this);
}

///OK button clicked (event handler for wxID_OK)
void CalibDialog::OnOKClick( wxCommandEvent& event ){
  bool datavalid = true;
  //Transfer data from the dialog window to our variables
  TransferDataFromWindow();
  
  //Grab MAC address from dialog window.
  MacAddr* macaddr = new MacAddr();
  char buf[18];
  strcpy(buf, mac.char_str());
  datavalid = datavalid && macaddr->macBuilder(buf, strlen(buf)); //Sanitizes MAC address

  if (*macptr != NULL){
    //Timer hasn't had a chance to save the last new intruder we set for calibration!
    //Ask user to wait 2 * update period
    //FIXME: This also occurs when the first and only calibration mac address allowed is present
    cout << "You already have a calibration MAC address! Limitations only allow one." << endl;
    //cout << "Please wait " << (((float) UPDATE_RATE) / 500) << " seconds before creating a new calibration MAC address." << endl;
    return; //No need to reset data fields, make the user click "Cancel" instead of closing.
  }
  
  //Note: MAC address has been sanitized by MacBuilder and spin controls don't allow illegal positions
  cout << "Setting the Following as Calibration" << endl;
  macaddr->print(&cout);
  cout << "X: " << x << " cm" << endl;
  cout << "Y: " << y << " cm" << endl;
  
  if (!datavalid){
    cout << "Data entered was invalid!" << endl;
    delete macaddr;
    macaddr = NULL;
    Init();
    return; //without closing dialog!
  }
  
  *macptr = macaddr;
  *locptr = new Location( cm_to_meters(x), cm_to_meters(y), 0 );
  *calibdialog = NULL; //Tells frame that dialog is closed
  cout << "OK was clicked and data passed back\n";
  Close();
  Destroy();
}


void CalibDialog::OnCancelClick( wxCommandEvent& event ){
  *calibdialog = NULL;
  Close();
  Destroy();
}


/*!
 * Control creation for CalibDialog
 */

void CalibDialog::CreateControls(){   
  //WORKAROUND: Since the spin controls only handle integers, use small units (cm)
  //FIXME still using pseudo-meters for the max and min values
  //NOTE: Set the max/min values for an Intruder location in location.h
  
  // A top-level sizer
  wxBoxSizer* topSizer = new wxBoxSizer(wxVERTICAL);
  this->SetSizer(topSizer);
  // A second box sizer to give more space around the controls
  wxBoxSizer* boxSizer = new wxBoxSizer(wxVERTICAL);
  topSizer->Add(boxSizer, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5);
  // A friendly message
  wxStaticText* descr = new wxStaticText( this, wxID_STATIC,
      _("Please select the intruder's location and the pattern it will follow."),
					  wxDefaultPosition, wxDefaultSize, 0 );
  boxSizer->Add(descr, 0, wxALIGN_LEFT|wxALL, 5);
  // Spacer
  boxSizer->Add(5, 5, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5);
  
  // Label for the MAC Address text control
  wxStaticText* macLabel = new wxStaticText ( this, wxID_STATIC,
      wxT("&MAC Address:"), wxDefaultPosition, wxDefaultSize, 0 );
  boxSizer->Add(macLabel, 0, wxALIGN_LEFT|wxALL, 5);
  // A text control for the MAC address
  wxTextCtrl* nameCtrl = new wxTextCtrl ( this, NRLID_MAC, wxT("FF:FF:FF:FF:FF:FF"), wxDefaultPosition, wxDefaultSize, 0 );
  boxSizer->Add(nameCtrl, 0, wxGROW|wxALL, 5);
  
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
void CalibDialog::SetDialogValidators(){
  FindWindow(NRLID_MAC)->SetValidator(
      wxTextValidator(wxFILTER_NONE, & mac));
  FindWindow(NRLID_X)->SetValidator(
      wxGenericValidator(& x));
  FindWindow(NRLID_Y)->SetValidator(
      wxGenericValidator(& y));
}


// Sets the help text for the dialog controls
void CalibDialog::SetDialogHelp(void){
  wxString macHelp = wxT("Enter intruder's MAC address (A.B.C.D.E.F)");
  wxString xHelp = wxT("Enter intruder's X coordinate (in cm)"); 
  wxString yHelp = wxT("Enter intruder's Y coordinate (in cm)");
  FindWindow(NRLID_MAC)->SetHelpText(macHelp);
  FindWindow(NRLID_MAC)->SetToolTip(macHelp);
  FindWindow(NRLID_X)->SetHelpText(xHelp);
  FindWindow(NRLID_X)->SetToolTip(xHelp);
  FindWindow(NRLID_Y)->SetHelpText(yHelp);
  FindWindow(NRLID_Y)->SetToolTip(yHelp);
}
