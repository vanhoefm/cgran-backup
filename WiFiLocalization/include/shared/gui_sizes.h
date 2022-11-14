/** gui_sizes.h: Size of the GUI, its picture, its max and min positions, etc.
 * 
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
#ifndef GUI_SIZES_H
#define GUI_SIZES_H

#include <stdint.h>

///Maximum and minimum allowed X/Y values
//#define NRL_MIN_LOCATION	-120
//#define NRL_MAX_LOCATION	 120
///Units are in centimeters
#define MIN_X_POS		0
#define MIN_Y_POS		0
#define MAX_X_POS		3000
#define MAX_Y_POS		3000
///Intruder velocity assumes (quickly) walking human
#define MIN_INTRUDER_VEL	-200
#define MAX_INTRUDER_VEL	200
///Size of the background image
#define GUI_IMAGE_WIDTH		871
#define GUI_IMAGE_HEIGHT	934
#define PIXELS_PER_METER	30
///Location of (0, 0, 0) on the background image
#define PIXEL_X_ZERO		111
#define PIXEL_Y_ZERO		74


using namespace std;

///Unit conversion functions
//Use cm to work around the fact spin controls only handle integers,
//yet we represent everything in meters
float cm_to_meters(int32_t cm);

//Overload cm_to_meters for int data type
float cm_to_meters(int cm);

int32_t meters_to_cm(float meters);

int32_t meters_to_x_pixels(float meters);

float x_pixels_to_meters(int32_t pixels);
  
int32_t meters_to_y_pixels(float meters);
  
float y_pixels_to_meters(int32_t pixels);

#endif
