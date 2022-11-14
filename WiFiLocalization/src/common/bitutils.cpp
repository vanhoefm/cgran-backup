/** bitutils.cpp: Data type Utilities for the NRLMQP solver
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

#include <iostream>
#include <arpa/inet.h> //BUG: Does this break cross-platform ability?
#include <string.h>

#include "bitutils.h"

using namespace std;

enum ByteOrderStatus { ORDER_UNDEF, ORDER_SAME, ORDER_DIFF };

union FloatSplitter {
  float f;
  int32_t i;
  char c[4];
};

union NRL64BitUnion_t {
  double mydouble;
  uint64_t myint64_t;
  int32_t half[2];
  char bytes[8];
};

/* ---- Globals ----- */
ByteOrderStatus same_byteorder;


// Function for checking to see that all data types are big enough
void checkDataSizeErrors(void){
  if (sizeof(char) != 1)
    throw "Error: Char is not 8 bits!\n";
  if (sizeof(float) != 4)
    throw "Error: Float is not 32 bits!\n";
  if (sizeof(double) != 8)
    throw "Error: Double is not 64 bits!\n";
}

//Initialization function that MUST be called on startup for NRL byte-swapping functions to work
void initUtils(void){
  checkDataSizeErrors();
  //Initialization for 64-bit data type swapping
  int32_t q = 314159;
  if (q == htonl(q))
    same_byteorder = ORDER_SAME;
  else
    same_byteorder = ORDER_DIFF;
  
#ifdef PRINT_BYTEORDER
  if (same_byteorder == ORDER_SAME)
    cout << "Host is Big-Endian" << endl;
  else
    cout << "Host is Little-Endian" << endl;
#endif
  return;
}

//Convert uint64_t to network-byte-order 64-bit value
void nrl_htonul(int64_t val, char* buf){
  if (same_byteorder == ORDER_UNDEF)
    throw "ERROR: Cannot perform htond without knowing byte order";
  
  //Swap each half of the double around
  NRL64BitUnion_t item;
  item.myint64_t = val;
  if (same_byteorder == ORDER_DIFF){
    int64_t temp = htonl(item.half[0]);
    item.half[0] = htonl(item.half[1]);
    item.half[1] = temp;
  }
  memcpy(buf, item.bytes, 8);
}

int64_t nrl_ntohul(char* buf){
  if (same_byteorder == ORDER_UNDEF)
    throw "ERROR: Cannot perform ntohd without knowing byte order";
  
  //Swap each half of the double around
  NRL64BitUnion_t item;
  memcpy(item.bytes, buf, 8);
  if (same_byteorder == ORDER_DIFF){
    int64_t temp = htonl(item.half[0]);
    item.half[0] = htonl(item.half[1]);
    item.half[1] = temp;
  }
  return item.myint64_t;
}

void nrl_htonf(float val, char* buf){
  FloatSplitter helper;
  helper.f = val;
  helper.i = htonl(helper.i);
  memcpy(buf, helper.c, 4);
}
  
float nrl_ntohf(char* buf){
  FloatSplitter helper;
  memcpy(helper.c, buf, 4);
  helper.i = ntohl(helper.i);
  return helper.f;
}

//Convert double to network-byte-order 64-bit value
void nrl_htond(double val, char* buf){
  if (same_byteorder == ORDER_UNDEF)
    throw "ERROR: Cannot perform htond without knowing byte order";
  
  //Swap each half of the double around
  NRL64BitUnion_t item;
  item.mydouble = val;
  if (same_byteorder == ORDER_DIFF){
    int64_t temp = htonl(item.half[0]);
    item.half[0] = htonl(item.half[1]);
    item.half[1] = temp;
  }
  memcpy(buf, item.bytes, 8);
}

double nrl_ntohd(char* buf){
  if (same_byteorder == ORDER_UNDEF)
    throw "ERROR: Cannot perform ntohd without knowing byte order";
  
  //Swap each half of the double around
  NRL64BitUnion_t item;
  memcpy(item.bytes, buf, 8);
  if (same_byteorder == ORDER_DIFF){
    int64_t temp = htonl(item.half[0]);
    item.half[0] = htonl(item.half[1]);
    item.half[1] = temp;
  }
  return item.mydouble;
}
