/*
 * Copyright 2010 Communications Engineering Lab, KIT
 *
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#include <cstdio>
#include <stdexcept>
#include <cppunit/TestAssert.h>
#include <gr_complex.h>
#include <qa_specest_arburg.h>
#include <specesti_arburg.h>
#include <iostream>
#include <cmath>

using std::exp;
using std::abs;
using std::cout;
using std::endl;

const int ar_proc_len = 256;
const int ar_proc_order = 4;
const float ar_process[ar_proc_len*2] = {
	1.4310, -0.3066,
	2.3536, -0.9656,
	2.6211, -1.6518,
	2.3034, -1.3116,
	2.0013,  0.6699,
	0.3519,  3.8874,
	-3.3812, 7.4823,
	-7.6895, 9.6972,
	-8.9602, 7.0478,
	-5.9128,-2.8789,
	0.8732,-15.3469,
	7.4119,-22.9530,
	9.7694,-18.9880,
	6.9671,-2.9939,
	1.0988, 18.9369,
	-3.6737, 34.4581,
	-4.1556, 32.2055,
	-1.4540, 10.7854,
	1.2396,-18.8327,
	0.6621,-39.4158,
	-3.8499,-38.6139,
	-7.8648,-17.2044,
	-6.4302, 12.6777,
	1.3515, 33.5593,
	11.5656, 33.6270,
	17.4025, 15.4283,
	13.7454,-8.5046,
	1.9575,-24.1417,
	10.8231,-23.7297,
	16.7723,-10.5482,
	13.0526, 5.8286,
	-3.1100, 15.3772,
	7.4885, 14.8894,
	12.2439, 8.1645,
	9.0544, 1.0704,
	-0.2944,-3.4651,
	-9.0227,-6.4728,
	10.4626,-9.5586,
	-3.6472,-12.2162,
	6.0801,-11.5650,
	9.4981,-4.0649,
	3.4508, 9.0515,
	-8.7140, 21.8845,
	19.2602, 25.4466,
	19.5270, 15.3183,
	-7.4965,-5.3381,
	10.9481,-25.6783,
	25.2469,-32.7227,
	26.7339,-21.0964,
	13.0857, 2.9365,
	-8.5498, 26.8052,
	25.7815, 37.7338,
	27.9848, 29.0029,
	13.4110, 5.1540,
	9.7372,-21.1807,
	27.4513,-35.5135,
	28.8425,-29.4464,
	13.9589,-8.0639,
	-9.0257, 16.0057,
	27.2919, 30.6175,
	31.4544, 29.3729,
	19.9534, 15.5743,
	1.1527,-2.3653,
	21.5482,-17.0954,
	30.4895,-25.5414,
	23.2192,-26.2681,
	4.1280,-17.8631,
	16.2913,-0.9747,
	27.0466, 19.5638,
	22.8010, 34.1825,
	-7.5832, 33.8352,
	9.1074, 14.8073,
	17.0148,-15.9612,
	14.0197,-42.8637,
	4.5944,-49.1105,
	-4.8554,-28.2850,
	-9.6085, 9.2433,
	-9.7944, 42.5767,
	-7.5682, 52.1452,
	-4.9781, 31.9000,
	-0.4770,-5.5890,
	7.4238,-37.8724,
	14.3288,-46.5004,
	14.9138,-27.7316,
	5.7220, 5.4319,
	10.0506, 32.6572,
	23.0979, 39.3120,
	23.5330, 24.9450,
	-9.0988, 1.0376,
	13.6646,-17.0466,
	30.9490,-21.1233,
	31.1980,-13.6821,
	13.3288,-2.2684,
	12.5264, 5.2112,
	30.5561, 6.2367,
	29.8474, 4.6307,
	12.0690, 5.0478,
	9.6322, 9.0180,
	22.9237, 12.9184,
	21.7015, 10.6222,
	9.3396,-1.1856,
	-4.5837,-17.4015,
	11.7555,-28.2480,
	10.8916,-24.4600,
	-6.3945,-4.3929,
	-3.1969, 22.3499,
	-1.4541, 40.2538,
	0.6530, 38.3435,
	5.3226, 16.4807,
	11.5192,-14.2581,
	14.4291,-37.1521,
	8.7124,-39.6230,
	-5.4829,-21.8040,
	20.7514, 5.0873,
	27.6453, 27.1782,
	19.7068, 34.3613,
	0.3606, 25.3493,
	21.8368, 5.7509,
	32.3954,-14.6036,
	24.7448,-26.2847,
	2.2742,-26.0359,
	21.9784,-15.5346,
	34.9064,-0.0581,
	28.7074, 13.5670,
	-4.9444, 20.4800,
	24.1003, 17.6968,
	41.2230, 5.3632,
	35.6707,-11.6508,
	9.1771,-25.2628,
	22.1425,-28.2930,
	38.8634,-18.9320,
	30.9584,-0.5203,
	-5.2262, 17.8408,
	21.0994, 27.2149,
	31.5234, 23.8132,
	21.1365, 9.8556,
	-1.3592,-7.7348,
	20.8647,-21.1297,
	26.1002,-25.7534,
	15.8284,-20.2408,
	0.4495,-5.9854,
	12.0044, 13.0175,
	13.4550, 28.5235,
	7.7107, 30.8056,
	1.6444, 16.5423,
	0.1200,-7.7917,
	1.0469,-29.3147,
	-1.0480,-36.6136,
	-8.3996,-23.8981,
	16.4504, 1.2872,
	16.3570, 24.1858,
	-3.9963, 31.3402,
	16.2982, 19.4015,
	32.3522,-2.6508,
	32.5456,-20.3333,
	13.5961,-24.0058,
	16.1576,-13.1983,
	40.9904, 3.9745,
	45.4703, 16.8080,
	24.1876, 18.1106,
	12.4462, 9.4298,
	43.3584,-1.5839,
	49.8775,-6.0027,
	27.0704,-1.4925,
	12.1249, 6.6573,
	44.4599, 9.6387,
	50.7506, 1.9153,
	27.9143,-12.6605,
	9.9861,-23.3596,
	40.4215,-20.4657,
	47.6341,-2.4568,
	29.9793, 20.5212,
	0.5586, 32.6560,
	24.1526, 24.0527,
	32.8887,-1.4276,
	25.1566,-27.6119,
	-8.3047,-37.8054,
	8.3969,-24.9353,
	16.9110, 3.3284,
	14.9537, 30.3347,
	5.7759, 39.6700,
	-3.6369, 25.6956,
	-6.9529,-3.6387,
	-4.0552,-31.6993,
	0.8876,-41.8909,
	3.1992,-28.4243,
	0.3688, 0.7622,
	-4.7904, 27.9128,
	-7.4925, 38.7789,
	-3.3352, 29.0599,
	6.2686, 6.5573,
	15.7156,-15.6791,
	17.2700,-27.1011,
	7.9230,-24.0290,
	-8.0701,-10.3345,
	22.5922, 4.8138,
	27.2434, 12.8615,
	17.4211, 11.1623,
	3.1873, 3.1250,
	22.9731,-4.1846,
	29.7741,-6.0138,
	19.3835,-2.9361,
	-2.6326,-0.1426,
	22.6581,-0.7461,
	29.0448,-3.4409,
	17.4560,-4.3857,
	4.0325,-0.1217,
	21.6524, 7.3171,
	23.8460, 12.4985,
	9.6375, 10.5970,
	10.9514, 1.4037,
	23.4015,-10.6162,
	18.6629,-17.8529,
	-0.0989,-15.2613,
	18.6009,-4.2023,
	24.4013, 8.1370,
	14.0274, 15.2079,
	-4.7243, 13.3267,
	18.5261, 4.1913,
	18.1706,-7.2293,
	-5.7607,-14.8260,
	9.0800,-16.8572,
	15.4527,-13.8653,
	9.1331,-7.7126,
	-4.1298,-0.2090,
	13.8054, 7.4431,
	12.5094, 13.4579,
	-0.9024, 16.7357,
	13.0998, 14.4587,
	19.0230, 4.8008,
	11.9921,-10.9766,
	-3.9570,-26.0289,
	18.0829,-31.5947,
	20.2843,-21.5776,
	-9.1063, 2.5010,
	8.0221, 29.6231,
	20.1530, 43.6857,
	19.7075, 33.1382,
	8.5339, 2.3426,
	-5.9032,-29.8051,
	15.6607,-43.5495,
	17.5384,-31.7484,
	11.6454,-2.1032,
	-0.8069, 26.7304,
	10.0212, 37.4636,
	16.6764, 25.9419,
	16.5986, 0.7517,
	9.8190,-23.4467,
	-1.0703,-33.0681,
	11.4211,-23.6324,
	16.6591,-2.0755,
	12.4093, 18.5888,
	-0.9193, 27.8319,
	9.7836, 23.0118,
	12.1677, 8.3128,
	4.5613,-7.4870
};

const float ar_coeff[(ar_proc_order+1)*2] = {
	1.0000,  0,
	-2.7733, 0.0057,
	3.8177, -0.0177,
	-2.6493, 0.0232,
   	0.9132,  0.0136
};


void print_error(const float *expected_result, gr_complex *real_data, int n) {
	printf("\n/ Index        | Calculated | Expected   | Factor         \\\n");
	printf("|--------------+------------+------------+----------------|\n");
	for (int i = 1; i < n; i++) {
		printf("| Coeff %d real | %+6.4f    | %+6.4f    | %+6.4f        |\n",
			   	i, real_data[i].real(), expected_result[(2*i)],
				real_data[i].real() / expected_result[(2*i)]);
		printf("|         imag | %+6.4f    | %+6.4f    | %+6.4f        |\n",
			   	real_data[i].imag(), expected_result[(2*i)+1],
				real_data[i].imag() / expected_result[(2*i)+1]);
	}
	printf("\\--------------+------------+------------+----------------/\n");
	//for (int i = 1; i < n; i++) {
		//std::cout << real_data[i] << std::endl;
	//}
}

void
qa_specest_arburg::t1()
{
	CPPUNIT_ASSERT_THROW( specesti_arburg *AR = new specesti_arburg(23, 42), std::invalid_argument );
	CPPUNIT_ASSERT_THROW( specesti_arburg *AR = new specesti_arburg(0, 42), std::invalid_argument );
	CPPUNIT_ASSERT_THROW( specesti_arburg *AR = new specesti_arburg(23, 0), std::invalid_argument );
}


/**
 * Compare coefficients to precalculated results.
 */
void
qa_specest_arburg::t2()
{
	// tbw
	// In fact, I wrote this once and it simply didn't work.
	// I always got other results than in the Python domain.
	// This nearly cost my my sanity, so I gave up for now and run all
	// the tests in Python. The data's all here, so if you want to give
	// it a bash...
}

