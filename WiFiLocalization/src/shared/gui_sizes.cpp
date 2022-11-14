/** gui_sizes.cpp: Size of the GUI, its picture, its max and min positions, etc.
 * This file contains conversion function implementations
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
#include "gui_sizes.h"

///Unit conversion functions
//Use cm to work around the fact spin controls only handle integers,
//yet we represent everything in meters

/* Use this only if your machine has 16-bit int (unlikely)
float cm_to_meters(int32_t cm){
  return ((float) cm) * 100;
}
*/

//Overload cm_to_meters for int data type
float cm_to_meters(int cm){
  return ((float) cm) * 100;
}

int32_t meters_to_cm(float meters){
  return (int32_t) (meters / 100);
}

int32_t meters_to_x_pixels(float meters){
  return (int32_t) (meters * PIXELS_PER_METER + PIXEL_X_ZERO); 
}

float x_pixels_to_meters(int32_t pixels){
  return ((float) (pixels - PIXEL_X_ZERO)) / PIXELS_PER_METER;
}
  
int32_t meters_to_y_pixels(float meters){
  return (int32_t) (meters * PIXELS_PER_METER + PIXEL_Y_ZERO); 
}

float y_pixels_to_meters(int32_t pixels){
  return ((float) (pixels - PIXEL_Y_ZERO)) / PIXELS_PER_METER;
}

