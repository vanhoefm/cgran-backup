/*
#<license>
 * Copyright (c) 2006 BBN Technologies Corp.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of BBN Technologies nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY BBN TECHNOLOGIES AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL BBN TECHNOLOGIES OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Effort sponsored in part by the Defense Advanced Research Projects
 * Agency (DARPA) and the Department of the Interior National Business
 * Center under agreement number NBCHC050166.
#</license>
 */

/* $Id: usrp_data2xplot.c,v 1.1 2006/06/24 02:10:13 jmmikkel Exp $ */

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/endian.h>

#define USRP_READ_LEN 16384
#define NSAMPLES (USRP_READ_LEN >> 2)

int main(int argc, char *argv[]) {
	FILE *in, *out;
	size_t count;
	unsigned short buf[NSAMPLES];
	unsigned short last = 0;
	unsigned int i;
	int n;

	if (argc != 3) {
		printf("Usage: %s <usrp output> <xplot file>\n", argv[0]);
		return 1;
	}
	in = fopen(argv[1], "rb");
	if (!in) {
		fprintf(stderr, "Error opening input file %s: %s\n",
			argv[1], strerror(errno));
		return -1;
	}
	out = fopen(argv[2], "wb");
	if (!out) {
		fprintf(stderr, "Error opening output file %s: %s\n",
			argv[2], strerror(errno));
		fclose(in);
		return -1;
	}

	fprintf(out, "unsigned unsigned\ntitle\n%s\n", argv[1]);
	fprintf(out, "dot 0 0\n");

	count = fread((void*)buf, sizeof(unsigned short), NSAMPLES, in);
	n = 0;
	while (count > 0) {
		if (!n) {
			last = le16toh(buf[0]-1);
		}
		for (i = 0; i < count; i++) {
			/* note: USRP data is little-endian */
			if (!(last+1 == le16toh(buf[i])
			      || (last == 0xFFFF && le16toh(buf[i]) == 0))) {

				last++;
				if (last > buf[i]) {
					while (last < 0xFFFF) {
						fprintf(out,
							"dot %d 2 green\n",
							n);
						n++;
						last++;
					}
					fprintf(out,
						"dot %d 2 green\n",
						n);
					n++;
					last = 0;
				}
				while (last < buf[i]) {
					fprintf(out,
						"dot %d 2 green\n",
						n);
					n++;
					last++;
				}
			}
			fprintf(out, "dot %d 1\n", n);
			last = buf[i];
			n++;
		}
		count = fread((void*)buf, 2, NSAMPLES, in);
	}
	fprintf(out, "go\n");
	fclose(in);
	fclose(out);
}
