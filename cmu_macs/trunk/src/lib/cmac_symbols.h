/* -*- c++ -*- */
/*
 * Copyright 2007 Free Software Foundation, Inc.
 * 
 * This file is part of GNU Radio
 * 
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 * 
 * GNU Radio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef INCLUDED_CMAC_SYMBOLS_H
#define INCLUDED_CMAC_SYMBOLS_H

#include <pmt.h>

// CS
static pmt_t s_cmd_set_address = pmt_intern("cmd-set-address");
static pmt_t s_cmd_carrier_sense_enable = pmt_intern("cmd-carrier-sense-enable");
static pmt_t s_cmd_carrier_sense_threshold = pmt_intern("cmd-carrier-sense-threshold");
static pmt_t s_cmd_carrier_sense_deadline = pmt_intern("cmd-carrier-sense-deadline");
static pmt_t s_cmd_carrier_sense_disable = pmt_intern("cmd-carrier-sense-disable");
static pmt_t s_cmd_set_tx_retries = pmt_intern("cmd-set-tx-retries");
static pmt_t s_cmd_rx_enable = pmt_intern("cmd-rx-enable");
static pmt_t s_cmd_rx_disable = pmt_intern("cmd-rx-disable");
static pmt_t s_response_set_address = pmt_intern("response-set-address");
static pmt_t s_response_carrier_sense_enable = pmt_intern("response-carrier-sense-enable");
static pmt_t s_response_carrier_sense_treshold = pmt_intern("response-carrier-sense-threshold");
static pmt_t s_response_carrier_sense_deadline = pmt_intern("response-carrier-sense-deadline");
static pmt_t s_response_carrier_sense_disable = pmt_intern("response-carrier-sense-disable");
static pmt_t s_response_set_tx_retries = pmt_intern("response-set-tx-retries");
static pmt_t s_response_rx_enable = pmt_intern("response-rx-enable");
static pmt_t s_response_rx_disable = pmt_intern("response-rx-disable");

#endif // INCLUDED_CMAC_SYMBOLS_H
