/* -*- c++ -*- */
/*
 * Copyright 2011 Anton Blad.
 * 
 * This file is part of OpenRD
 * 
 * OpenRD is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 * 
 * OpenRD is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "field_coder.h"
#include "golay.h"
#include "rm_2_6.h"
#include <cstring>
#include <iostream>

using namespace std;

//#define FIELD_CODE_EVAL

#ifdef FIELD_CODE_EVAL
static int num_logged = 0;
static int num_errors[6] = {0,0,0,0,0,0};
#endif

field_coder::field_coder(field_code_type code) :
	d_code(code)
{
}

field_coder::~field_coder()
{
}

int field_coder::codeword_length() const
{
	return codeword_length(d_code);
}

int field_coder::codeword_length(field_code_type code)
{
	switch(code)
	{
	case FIELD_CODE_32_78:
		return 78;
	case FIELD_CODE_REPEAT3_16_78:
		return 78;
	case FIELD_CODE_REPEAT7_11_78:
		return 78;
	case FIELD_CODE_GOLAY_12_78:
		return 78;
	case FIELD_CODE_GOLAY_R3_12_78:
		return 78;
	case FIELD_CODE_GOLAY_R2_12_78:
		return 78;
	case FIELD_CODE_R2_RM_11_78:
		return 78;
	default:
		throw "field_coder::codeword_length : invalid code";
	}
}

void field_coder::code(unsigned int val, char* codeword)
{
	switch(d_code)
	{
	case FIELD_CODE_32_78:
		for(int i = 0; i < 32; i++)
		{
			codeword[i] = (val >> (31-i)) & 0x1;
			codeword[39+i] = (val >> (31-i)) & 0x1;
		}
		for(int i = 32; i < 39; i++)
		{
			codeword[i] = 0;
			codeword[39+i] = 0;
		}
		break;
	case FIELD_CODE_REPEAT3_16_78:
		for(int i = 0; i < 16; i++)
		{
			codeword[i] = (val >> (15-i)) & 0x01;
			codeword[16+i] = (val >> (15-i)) & 0x01;
			codeword[32+i] = (val >> (15-i)) & 0x01;
		}
		for(int i = 48; i < 78; i++)
			codeword[i] = 0;
		break;
	case FIELD_CODE_REPEAT7_11_78:
		for(int i = 0; i < 11; i++)
		{
			codeword[i] = (val >> (10-i)) & 0x01;
			codeword[11+i] = (val >> (10-i)) & 0x01;
			codeword[22+i] = (val >> (10-i)) & 0x01;
			codeword[33+i] = (val >> (10-i)) & 0x01;
			codeword[44+i] = (val >> (10-i)) & 0x01;
			codeword[55+i] = (val >> (10-i)) & 0x01;
			codeword[66+i] = (val >> (10-i)) & 0x01;
		}
		for(int i = 77; i < 78; i++)
			codeword[i] = 0;
		break;
	case FIELD_CODE_GOLAY_12_78:
		{
			int c = golay_encode(val & 4095);
			for(int i = 0; i < 24; i++)
			{
				codeword[23-i] = c & 0x01;
				c >>= 1;
			}
			for(int i = 24; i < 78; i++)
				codeword[i] = 0;
		}
		break;
	case FIELD_CODE_GOLAY_R3_12_78:
		{
			int c = golay_encode(val & 4095);
			for(int i = 0; i < 24; i++)
			{
				codeword[23-i] = c & 0x01;
				codeword[24+23-i] = c & 0x01;
				codeword[48+23-i] = c & 0x01;
				c >>= 1;
			}
			for(int i = 72; i < 78; i++)
				codeword[i] = 0;
		}
		break;
	case FIELD_CODE_GOLAY_R2_12_78:
		{
			int c = golay_encode(val & 4095);
			for(int i = 0; i < 24; i++)
			{
				codeword[23-i] = c & 0x01;
				codeword[24+23-i] = c & 0x01;
				c >>= 1;
			}
			for(int i = 48; i < 78; i++)
				codeword[i] = 0;
		}
		break;
	case FIELD_CODE_R2_RM_11_78:
		{
			char msg[22];
			for(int i = 0; i < 11; i++)
			{
				msg[10-i] = msg[21-i] = (val & 0x01);
				val >>= 1;
			}
			rm_2_6_encode(msg, codeword);
			for(int i = 64; i < 78; i++)
				codeword[i] = 0;
		}
		break;
	default:
		throw "field_coder::code : invalid code";
	}
}

int field_coder::decode(const float* codeword, unsigned int* val)
{
	unsigned int v = 0;
	unsigned int d;
	int status = 0;

	switch(d_code)
	{
	case FIELD_CODE_32_78:
		v = 0;
		for(int i = 0; i < 32; i++)
			v = (v << 1) + (codeword[i] >= 0.0);
		break;
	case FIELD_CODE_REPEAT3_16_78:
		v = 0;
		for(int i = 0; i < 16; i++)
			v = (v << 1) + ((codeword[i]+codeword[16+i]+codeword[32+i]) >= 0.0);
		break;
	case FIELD_CODE_REPEAT7_11_78:
		v = 0;
		for(int i = 0; i < 11; i++)
			v = (v << 1) + ((codeword[i]+codeword[11+i]+codeword[22+i]+codeword[33+i]+codeword[44+i]+codeword[55+i]+codeword[66+i]) >= 0.0);
		break;
	case FIELD_CODE_GOLAY_12_78:
		d = 0;
		for(int i = 0; i < 24; i++)
			d = (d << 1) + (codeword[i] >= 0.0);
		v = golay_decode(d);
#ifdef FIELD_CODE_EVAL
		{
			int t = golay_numchanged(v, d);
			t = (t < 6 ? t : 5);
			if(t < 6)
				num_errors[t]++;
			num_logged++;
			if(num_logged >= 1000)
			{
				std::cerr << "field code errors:";
				for(int i = 0; i < 6; i++)
				{
					std::cerr << " " << i << ":" << num_errors[i];
					num_errors[i] = 0;
				}
				num_logged = 0;
				std::cerr << std::endl;
			}
		}
#endif
		break;
	case FIELD_CODE_GOLAY_R3_12_78:
		d = 0;
		for(int i = 0; i < 24; i++)
			d = (d << 1) + (codeword[i]+codeword[24+i]+codeword[48+i] >= 0.0);
		v = golay_decode(d);
#ifdef FIELD_CODE_EVAL
		{
			int t = golay_numchanged(v, d);
			t = (t < 6 ? t : 5);
			if(t < 6)
				num_errors[t]++;
			num_logged++;
			if(num_logged >= 1000)
			{
				std::cerr << "field code errors:";
				for(int i = 0; i < 6; i++)
				{
					std::cerr << " " << i << ":" << num_errors[i];
					num_errors[i] = 0;
				}
				num_logged = 0;
				std::cerr << std::endl;
			}
		}
#endif
		break;
	case FIELD_CODE_GOLAY_R2_12_78:
		d = 0;
		for(int i = 0; i < 24; i++)
			d = (d << 1) + (codeword[i] >= 0.0);
		v = golay_decode(d);
		d = 0;
		for(int i = 0; i < 24; i++)
			d = (d << 1) + (codeword[24+i] >= 0.0);
		if(v != golay_decode(d))
			v = 0xffffffff;
		break;
	case FIELD_CODE_R2_RM_11_78:
		{
			char x[64];
			char msg[22];
			for(int i = 0; i < 64; i++)
				x[i] = (codeword[i] >= 0.0);
			rm_2_6_decode(x, msg);
			for(int i = 0; i < 11; i++)
			{
				if(msg[i] != msg[11+i])
					status = 1;
				v = (v << 1) | (msg[i] & 0x01);
			}
		}
		break;
	default:
		throw "field_coder::decode : invalid code";
	}

	*val = v;

	return status;
}

