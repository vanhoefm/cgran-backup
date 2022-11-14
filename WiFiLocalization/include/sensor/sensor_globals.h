/** sensor_globals.h: Defined parameters for NRL Localization Sensor
 * 
 * These are intended to be adjusted by the user at compile time
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

#ifndef NRL_SENSOR_GLOBALS_H
#define NRL_SENSOR_GLOBALS_H

#include "atomic.h"
#include "location.h"
#include <stdint.h>

//Shutoff switch for testing without solver connection
#define TCP_SENDING_ENABLED 	0

//File for runtime-adjusted parameters (such as sensor's location)
#define NRL_PARAMS_FILE_NAME	"../bbn/nrl_solver_config.txt"

#define DEFAULT_PORT		5749
//The maximum number of packets that can be in the buffer waiting to be sent
#define BBNSENDER_PKTBUF_SIZE 	10

//Time between thread polls, in milliseconds
#define UPDATE_RATE 	100

extern Atomic* istimetodie;
extern Location myloc;
extern char senderhost[];
extern int senderport;

#endif