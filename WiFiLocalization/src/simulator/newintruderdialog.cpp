/** newintruderdialog.cpp: Dialog window for creating a new intruder
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

#include "newintruderdialog.h"
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
  NRLID_X = 10002,
  NRLID_Y = 10003,
  NRLID_VEL,
  NRLID_STEPS,
  NRLID_PATTERN = 10006,
  NRLID_SECS = 10005,
  ID_RESET = 10004
};

/*!
 * NewIntruderDialog event table definition
 */

BEGIN_EVENT_TABLE( NewIntruderDialog, wxDialog )
  EVT_BUTTON( ID_RESET, NewIntruderDialog::OnResetClick )
  EVT_BUTTON( wxID_HELP, NewIntruderDialog::OnHelpClick )
  EVT_BUTTON( wxID_CANCEL, NewIntruderDialog::OnCancelClick )
  EVT_BUTTON( wxID_OK, NewIntruderDialog::OnOKClick )
END_EVENT_TABLE()

//BUG: Clicking on Help while this is open doesn't show whether the dialog or the main help gets displayed. This may not matter, though.
 
/// Initialisation
void NewIntruderDialog::Init(void){
  mac = _("aa:bb:cc:dd:ee:ff"); //default Mac address
  x = 0;
  y = 0;
  pattern = ID_SquarePattern1;
  //NOTE: Velocities are only relevant for SquarePattern1
  patternvelocity = Location(0, 0, 0);
  patternvel = 0;
  numsteps = 0; 
  numsecs = 30;
}


//Call this function immediately after the constructor
//If it returns a negative value, delete the object
bool NewIntruderDialog::Create(wxWindow* parent, NRLApp* theapp, NewIntruderDialog** dialog, Intruder** newintruderptr){
  //Initialize the dialog base class
  SetExtraStyle(wxDIALOG_EX_CONTEXTHELP);
  if (!wxDialog::Create(parent, -1, _("New Intruder Information")))
    return false;
  
  app = theapp;
  intruderptr = newintruderptr;
  newintruderdialog = dialog;
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
void NewIntruderDialog::OnResetClick(wxCommandEvent& event){
  Init();
  TransferDataToWindow();
}

/*!
 * wxEVT_COMMAND_BUTTON_CLICKED event handler for wxID_HELP
 */
//FIXME: Use something more computationally efficient than a message box
void NewIntruderDialog::OnHelpClick( wxCommandEvent& event ){
  
  wxString helpText =
    wxT("Please enter some information about the new intruder.\n")
    wxT("X is the horizontal position in cm, Y is the vertical position in cm\n")
    wxT("Patterns are pre-determined scripts that dictate the intruder's behavior.\n");

  wxMessageBox(helpText,
    wxT("New Intruder Dialog Help"),
    wxOK|wxICON_INFORMATION, this);
}

///OK button clicked (event handler for wxID_OK)
void NewIntruderDialog::OnOKClick( wxCommandEvent& event ){
  bool datavalid = true;
  //Transfer data from the dialog window to our variables
  TransferDataFromWindow();
  
  int64_t totsteps = numsecs * (1000 / UPDATE_RATE);
  int32_t tempp = cm_to_meters(patternvel);
  patternvelocity = Location(tempp, -tempp, 0); //Go northeast 
  
  IntruderPattern* patternobj;
  NRLPatternID pattern2 = (NRLPatternID) pattern;
  if (pattern2 == ID_SquarePattern1)
    patternobj = new SquarePattern1(patternvelocity, numsteps, totsteps);
  else {
    cout << "Warning: Pattern selected does not match any known patterns. Unable to create intruder.\n";
    return;
  }
  
  //Grab MAC address from dialog window.
  MacAddr* macaddr = new MacAddr();
  char buf[18];
  strcpy(buf, mac.char_str());
  datavalid = datavalid && macaddr->macBuilder(buf, strlen(buf));

  if (*intruderptr != NULL){
    //Timer hasn't had a chance to save the last new intruder we made! Ask user to wait 2 * update period
    cout << "Please wait " << (((float) UPDATE_RATE) / 500) << " seconds before creating a new intruder." << endl;
    return;
  }
  
  //Sanitize intruder data or return //FIXME: This just prints it
  cout << "New Intruder information" << endl;
  macaddr->print(&cout);
  cout << "X: " << x << " cm" << endl;
  cout << "Y: " << y << " cm" << endl;
  cout << "Velocity: " << patternvel << endl;
  cout << "NumSteps: " << numsteps << endl;
  cout << "Total Steps: " << numsecs << endl;
  cout << "Using SquarePattern1? " << (pattern == ID_SquarePattern1) << endl;
    
  if (!datavalid){
    cout << "Data entered was invalid!" << endl;
    //delete *intruderptr;
    //*intruderptr == NULL;
    //cout << "Successful delete intruderptr" << endl;
    Init();
    return; //without closing dialog!
  }
  
  *intruderptr = new Intruder(macaddr, new Location(x, y, 0), patternobj, app);
  *newintruderdialog = NULL; //Tells frame that dialog is closed
  cout << "OK was clicked and data passed back\n";
  Close();
  Destroy();
}


void NewIntruderDialog::OnCancelClick( wxCommandEvent& event ){
  *newintruderdialog = NULL;
  Close();
  Destroy();
}


/*!
 * Control creation for NewIntruderDialog
 */

void NewIntruderDialog::CreateControls(){   
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
      wxSP_ARROW_KEYS, MIN_X_POS, MAX_X_POS, 0 );
  locbox->Add(xSpin, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);
  // Label for the Y control
  wxStaticText* yLabel = new wxStaticText ( this, wxID_STATIC,
      wxT("&Y:"), wxDefaultPosition, wxDefaultSize, 0 );
  locbox->Add(yLabel, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);
  // A spin control for the Y coordinate
  wxSpinCtrl* ySpin = new wxSpinCtrl ( this, NRLID_Y,
      wxEmptyString, wxDefaultPosition, wxSize(60, -1),
      wxSP_ARROW_KEYS, MIN_Y_POS, MAX_Y_POS, 0 );
  locbox->Add(ySpin, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);
  // Label for the pattern control
  wxStaticText* patternLabel = new wxStaticText ( this, wxID_STATIC,
      wxT("&Pattern:"), wxDefaultPosition, wxDefaultSize, 0 );
  locbox->Add(patternLabel, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);
  
  ///Additional patterns that are available go here

  // Create the pattern choice control
  wxString patternStrings[] = {
      wxT("Square Pattern 1")
  };

  wxChoice* patternChoice = new wxChoice ( this, NRLID_PATTERN,
      wxDefaultPosition, wxSize(80, -1), WXSIZEOF(patternStrings),
	  patternStrings, 0 );
  patternChoice->SetStringSelection(wxT("Square Pattern 1"));
  locbox->Add(patternChoice, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);
  
  // Spacer
  boxSizer->Add(5, 5, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5);
  
  // A horizontal box sizer to contain Velocity and NumSteps
  // FIXME: This only should be visible when Square Pattern 1 is selected
  wxBoxSizer* sp1box = new wxBoxSizer(wxHORIZONTAL);
  boxSizer->Add(sp1box, 0, wxGROW|wxALL, 5);
  // Label for the velocity control
  wxStaticText* velLabel = new wxStaticText ( this, wxID_STATIC,
      wxT("&Velocity:"), wxDefaultPosition, wxDefaultSize, 0 );
  sp1box->Add(velLabel, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);
  // A spin control for the velocity
  wxSpinCtrl* velSpin = new wxSpinCtrl ( this, NRLID_VEL,
      wxEmptyString, wxDefaultPosition, wxSize(60, -1),
      wxSP_ARROW_KEYS, MIN_INTRUDER_VEL, MAX_INTRUDER_VEL, 25 );
  sp1box->Add(velSpin, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);
  // Label for the Y control
  wxStaticText* stepsLabel = new wxStaticText ( this, wxID_STATIC,
      wxT("&Number of Steps:"), wxDefaultPosition, wxDefaultSize, 0 );
  sp1box->Add(stepsLabel, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);
  // A spin control for the Y coordinate
  wxSpinCtrl* stepsSpin = new wxSpinCtrl ( this, NRLID_STEPS,
      wxEmptyString, wxDefaultPosition, wxSize(60, -1),
      wxSP_ARROW_KEYS, 0, SQ1_MAX_NUM_STEPS, 30 );
  sp1box->Add(stepsSpin, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);
  
  // Label for the number of seconds control
  wxStaticText* secsLabel = new wxStaticText ( this, wxID_STATIC,
      wxT("&Seconds to Live:"), wxDefaultPosition, wxDefaultSize, 0 );
  sp1box->Add(secsLabel, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);
  // A spin control for the number of seconds to run for
  wxSpinCtrl* secsSpin = new wxSpinCtrl ( this, NRLID_SECS,
      wxEmptyString, wxDefaultPosition, wxSize(60, -1),
      wxSP_ARROW_KEYS, -1, MAX_TIME_TO_LIVE, 25 );
  sp1box->Add(secsSpin, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);
  
  
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
void NewIntruderDialog::SetDialogValidators(){
  FindWindow(NRLID_MAC)->SetValidator(
      wxTextValidator(wxFILTER_NONE, & mac));
  FindWindow(NRLID_X)->SetValidator(
      wxGenericValidator(& x));
  FindWindow(NRLID_Y)->SetValidator(
      wxGenericValidator(& y));
  FindWindow(NRLID_PATTERN)->SetValidator(
      wxGenericValidator((int*) &pattern)); //BUG? Using typecast to convert enum ptr to int ptr
  FindWindow(NRLID_VEL)->SetValidator(
      wxGenericValidator(& patternvel));
  FindWindow(NRLID_STEPS)->SetValidator(
      wxGenericValidator(& numsteps));
  FindWindow(NRLID_SECS)->SetValidator(
      wxGenericValidator(& numsecs));
}


// Sets the help text for the dialog controls
void NewIntruderDialog::SetDialogHelp(void){
  wxString macHelp = wxT("Enter intruder's MAC address (A.B.C.D.E.F)");
  wxString xHelp = wxT("Enter intruder's X coordinate (in cm)"); 
  wxString yHelp = wxT("Enter intruder's X coordinate (in cm)");
  wxString patternHelp = wxT("Select a pattern for the intruder to follow");
  wxString velHelp = wxT("Enter intruder's speed (in units)");
  wxString stepsHelp = wxT("Enter the distance the intruder will go before turning (in units)");
  wxString secsHelp = wxT("Enter the number of seconds this intruder should be around for");
  FindWindow(NRLID_MAC)->SetHelpText(macHelp);
  FindWindow(NRLID_MAC)->SetToolTip(macHelp);
  FindWindow(NRLID_X)->SetHelpText(xHelp);
  FindWindow(NRLID_X)->SetToolTip(xHelp);
  FindWindow(NRLID_Y)->SetHelpText(yHelp);
  FindWindow(NRLID_Y)->SetToolTip(yHelp);
  FindWindow(NRLID_PATTERN)->SetHelpText(patternHelp);
  FindWindow(NRLID_PATTERN)->SetToolTip(patternHelp);
  FindWindow(NRLID_SECS)->SetHelpText(secsHelp);
  FindWindow(NRLID_SECS)->SetToolTip(secsHelp);
  FindWindow(NRLID_VEL)->SetHelpText(velHelp);
  FindWindow(NRLID_VEL)->SetToolTip(velHelp);
  FindWindow(NRLID_STEPS)->SetHelpText(stepsHelp);
  FindWindow(NRLID_STEPS)->SetToolTip(stepsHelp);
}
