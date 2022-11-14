/** params.cpp: Implementation for parameter globals and container functions
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
#include "params.h"
#include <iostream>

using namespace std;

/*----------Global declarations----------*/

//ParamsWrapper* aggparams_w;


/*----------Utility functions-------------*/


///Initialize global variables and set default parameters
void initParams(bool useparamfile){
  
  if (useparamfile)
    cout << "Saved parameter file not implemented\n";
  
}

/*---------ParamsWrapper implementation----*/

ParamsWrapper::ParamsWrapper(Params* defaultparams){
  params = defaultparams;
  //initialize mutex?
}

ParamsWrapper::~ParamsWrapper(void){
  //Mutex obtain
  delete params;
  //mutex destroy
}

void ParamsWrapper::print(ostream* fs){
  params->print(fs);
}

void ParamsWrapper::printcsv(ostream* fs){
  params->printcsv(fs);
}

Params* ParamsWrapper::getParams(void){
  //get mutex via MutexLocker (allocated on stack)
  Params* ret = params->copy();
  return ret;
}

void ParamsWrapper::setParams(Params* newparams){
  //get mutex
  delete params;
  params = newparams;
  //release mutex
}

ParamsType ParamsWrapper::paramsType(void){
  //Mutex magic here
  return params->getType();
}

ParamsWrapper* ParamsWrapper::copy(void){
  Params* params2 = params->copy();
  return new ParamsWrapper(params2);
}


  
