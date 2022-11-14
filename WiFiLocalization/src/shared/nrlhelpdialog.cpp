/** NRLHelpDialog.cpp: A wxMessageDialog that clears its own pointer, used for "WHat Next?" help dialogs
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
#include "nrlhelpdialog.h"
#include <wx/sizer.h>
#include <wx/stattext.h>
#include <wx/button.h>

//Debug only
#include <iostream>

using namespace std;

/*!
 * event table definition
 */

BEGIN_EVENT_TABLE( NRLHelpDialog, wxDialog )
  EVT_BUTTON( wxID_OK, NRLHelpDialog::OnOKClick )
END_EVENT_TABLE()


void NRLHelpDialog::OnOKClick( wxCommandEvent& WXUNUSED ){
  //Clear main frame's pointer to this dialog and exit
  *mydialog = NULL;
  Close();
  Destroy();
} 


bool NRLHelpDialog::Create(NRLHelpDialog** dialog, wxString text, wxWindow* form){
  
  //cout << "Creating help dialog" << endl;
  
  if (! wxDialog::Create(form, -1, _("What Next?"), wxDefaultPosition, wxDefaultSize, wxOK, text))
    return false;
  
  //cout << "True" << endl;
  
  mydialog = dialog;
  
  // A top-level sizer
  wxBoxSizer* topSizer = new wxBoxSizer(wxVERTICAL);
  this->SetSizer(topSizer);
  
  // A second box sizer to give more space around the controls
  wxBoxSizer* boxSizer = new wxBoxSizer(wxVERTICAL);
  topSizer->Add(boxSizer, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5);

  // A friendly message
  wxStaticText* descr = new wxStaticText( this, wxID_STATIC,
			text, wxDefaultPosition, wxDefaultSize, 0 );
  boxSizer->Add(descr, 0, wxALIGN_LEFT|wxALL, 5);
  // Spacer
  boxSizer->Add(5, 5, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5);
  
  wxButton* ok = new wxButton ( this, wxID_OK, wxT("&OK"),
      wxDefaultPosition, wxDefaultSize, 0 );
  boxSizer->Add(ok, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5);
  
  //cout << "Sizers and such created successfully" << endl;
  
  // Fit the dialog to the minimum size dictated by the sizers
  GetSizer()->Fit(this);
  // Ensure that the dialog cannot be sized smaller than the minimum size
  GetSizer()->SetSizeHints(this);
  // Center the dialog
  Center();
  
  //cout << "Created successfully" << endl;
}
