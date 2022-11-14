/** bitutils.h: Data type Utilities for the NRLMQP solver
 * These don't fit into categories well (many are related to byte manipulation)
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

//Note: The implementation file has many elements that are normally found in .h files
//none of these are needed outside utils.cpp itself

#ifndef NRL_BITUTILS_H
#define NRL_BITUTILS_H

#include <stdint.h>

//Many classes that need bitutils will also need memset()
//so include the correct header
#include <string.h>

void initUtils(void);

void nrl_htonul(int64_t val, char* buf);
int64_t nrl_ntohul(char* buf);
void nrl_htonf(float val, char* buf);
float nrl_ntohf(char* buf);
void nrl_htond(double val, char* buf);
double nrl_ntohd(char* buf);

#endif
