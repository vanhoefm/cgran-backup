/** templates.h: Abstract classes that other things can be derived from
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

#ifndef NRL_TEMPLATES_H
#define NRL_TEMPLATES_H

#include <iostream>

using namespace std;


///Objects that can be printed to a std::stream
class Printable{
public:
  virtual void print(ostream* fs) = 0;
  virtual void printcsv(ostream* fs) = 0;
  virtual Printable* copy(void) = 0; //Return identical copy of this object
};


///Objects that can be serialized
///NOTE: Objects must have a constructor capable of unserialization
///in the form Serializable::Serializable(char* buf, int buflen)
///but this is not enforced
class Serializable{
public:
  virtual int serialize(char* buf, int buflen) = 0;
};

#endif
