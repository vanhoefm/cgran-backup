#include <stdio.h>
#include <unistd.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/*
 * 
 * CADENCE
 * 
 * A program to measure the cadence of DAB frames, and feed that data
 *   back into the 'gr_iqtx' flow-graph.
 * 
 * It simply looks for and measures the silence periods that "frame"
 *   the COFDM symbols that DAB/DAB+ use
 *
 * 
 * We expect a stream of complex_float (2 x float) input samples, which are
 *   baseband samples directly out of the 'gr_iqtx' flow-graph. We generally
 *   get this data via a named-pipe/FIFO that is setup by the gr_iqtx shell
 *   script.
 */

/*
 * Number of samples (at baseband rates out of the GR flow-graph)
 *   we try to read at once.  X2 because they're complex_float.
 */
#define NSAMPLES       5000

/*
 * The power ratio that constitutes "silence", based on the
 *   running average computed power levels.
 */
#define SILENCE_RATIO  100.0

void
write_rcvr_params (char *pname, char *parm, int pfd[]);

void
handle_pipe_input (int pfd[]);

int XMLRPC_PORT = 8081;

int
main (int argc, char **argv)
{
	
	/*
	 * Counter for post-detector (decimated) samples
	 */
	long long int samples = 0LL;
	
	/*
	 * Pipe FDs
	 */
	 int pfd[2];
	 
	/*
	 * Counter for input baseband samples
	 */
	long long int isamples = 0LL;
	
	/*
	 * Buffer for those samples
	 */
	float samplebuf[NSAMPLES*2];
	
	/*
	 * Tracking when a silence interval starts and ends
	 */
	long long int zend = -1LL;
	long long int zprevious;
	
	/*
	 * Tracking running-average cadence
	 */
	long long int avg_cadence, cadcnt = 0;
	int max_cadence = 0LL;
	int min_cadence = 99999999LL;
	
	/*
	 * The input (baseband) and post-detector sample rates
	 */
	int inrate, detrate;
	
	/*
	 * Alpha values for immediate post-detector, and longer-term
	 *   running average.
	 */
	double a, a2;
	
	/*
	 * Detector, and detector longer-term running average
	 */
	double detector = 0.0;
	double davg = 0.001;
	
	/*
	 * SNR estimator
	 */
	double snr_avg = 0.0;
	int snr_count = 0;
	
	int i, cnt;
	
	inrate = atoi(argv[1]);
	detrate = atoi(argv[2]);
	XMLRPC_PORT = atoi(argv[3]);
	
	/*
	 * Establish Alpha values for two low-pass filters
	 */
	a = 1.0/(double)(inrate/detrate);
	a *= 2.0;
	
	a2 = 1.0/(double)(inrate*2);
	
	zend = zprevious = -1LL;
	avg_cadence = 0L;
	
	
	/*
	 * We set up a pipe, and fork ourselves.
	 * 
	 * If we do the XMLRPC/Python calls directly out of the main read loop,
	 *   it takes too long, and we end up stalling the Gnu Radio pipeline
	 *   as a result, and causing an overrun on the TX every time we to
	 *   the XMLRPC thing.
	 * 
	 * By having a child process handle the system() calls, the main read
	 *   and analysis loop doesn't cause a stall.
	 */
	if (pipe (pfd))
	{
		exit (0);
	}
	
	/*
	 * If we're the child
	 */
	if (fork() == 0)
	{
		handle_pipe_input (pfd);
		exit(0);
	}
	
	/*
	 * Close down the reader side, we don't need it.
	 */
	close (pfd[0]);
	
	
	/*
	 * While more samples to read
	 */
	while ((cnt = fread (samplebuf, sizeof(float)*2, NSAMPLES, stdin)) > 0)
	{	
		/*
		 * Handle each of the samples
		 * They come in as floats, but they're actually complex_float, hence
		 *   the i += 2
		 */
		for (i = 0; (i < cnt*2); i += 2)
		{
			double deto;
			
			/*
			 * The instantaneous detector value
			 */
			deto = samplebuf[i]*samplebuf[i];
			deto += (samplebuf[i+1]*samplebuf[i+1]);
			
			/*
			 * Low-pass filter it a bit
			 */
			detector = (a * deto) + ((1.0 - a) * detector);
			
			/*
			 * Update the longer-term average
			 */
			davg = (a2 * deto) + ((1.0 - a2) * davg);
			
			/*
			 * We don't do any timing analysis until we've seen about
			 *   5 seconds of data, which gives us better average power
			 *   estimates against which to look for "silence".
			 *
			 * If it's time to process some detector output, do so.
			 *   We've effectively implemented a "KEEP_ONE_IN_N" here
			 *   for the post-detector processing.
			 */
			if ((isamples > (inrate*5)) &&
				((isamples % (inrate/detrate)) == 0))
			{
				
				/*
				 * If the short-term detector value
				 *   is smaller by a factor of at least
				 *   SILENCE_RATIO, conclude that it's "silence".
				 */
				if (detector < (davg/SILENCE_RATIO))
				{
					snr_avg += (davg/detector);
					snr_count++;
					
					if (zend == -1LL)
					{
						zend = samples;
						
						if (zprevious != -1LL)
						{
							int x;
							
							x = zend - zprevious;
							
							avg_cadence += x;
							cadcnt++;
							
							/*
							 * Update min, max
							 */
							if (x < min_cadence)
							{
								min_cadence = x;
							}
							else if (x > max_cadence)
							{
								max_cadence = x;
							}
						}
						
						/*
						 * Update previous silence time marker
						 */
						zprevious = zend;
					}
				}
				if (detector > (davg*0.75))
				{
					zend = -1LL;
				}
				
				/*
				 * Update post-detector samples counter
				 */
				samples++;
				
				/*
				 * Every two seconds (detrate * 2 samples)
				 * 
				 * Send calculated stats values back into the flow-graph
				 *   via XMLRPC.
				 */
				if (((samples % (detrate * 2)) == 0) &&
					cadcnt != 0)
				{
					double srate;
					char value[15];
					double snr;
					
					srate = (double)detrate;
					avg_cadence /= cadcnt;

					/*
					 * Average cadence
					 */
					sprintf (value, "%8.5f", (double)avg_cadence/srate);
					write_rcvr_params ("a1_value", value, pfd);

					/*
					 * Min cadence
					 */
					sprintf (value, "%8.5f", (double)min_cadence/srate);
					write_rcvr_params ("a2_value", value, pfd);
					
					/*
					 * Max cadence
					 */
					sprintf (value, "%8.5f", (double)max_cadence/srate);
					write_rcvr_params ("a3_value", value, pfd);
					
					/*
					 * SNR avg
					 */
					snr = snr_avg/snr_count;
					snr = 10.0*log10(snr);
					sprintf (value, "%7.2f dB", (double)snr);
					write_rcvr_params ("a4_value", value, pfd);
					snr_avg = 0.0;
					snr_count = 0;
					
					/*
					 * Reset max, min, cadence
					 */
					 cadcnt = 0;
					 avg_cadence = 0;
					 max_cadence = 0;
					 min_cadence = 9999999;
				}
			}
			
			/*
			 * Update input (baseband) sample counter
			 */
			isamples++;
		}
	}
	exit(0);
}

static int failed_writes = 0;

/*
 * Borrowed from "IRA"
 * 
 * Send XMLRPC commands back into the flow-graph that's listening on 8081
 */
void
write_rcvr_params (char *pname, char *parm, int pfd[])
{
	char sys_str[256];
	int r;
	
	sprintf (sys_str, 
		"python -c \"import xmlrpclib;xmlrpclib.Server('http://localhost:%d').set_%s('%s')\"\n",
		XMLRPC_PORT, pname, parm);
		
	r = write (pfd[1], sys_str, strlen(sys_str));
	if (r <= 0)
	{
		failed_writes++;
	}
}


/*
 * Handle input over the pipe from our parent.
 * 
 * Each line, we simply hand off to system() to do the XMLRPC/Python
 *   thing.
 * 
 * Since we don't do this very often, and we're a separate process from the main
 *   data reader, the fact that "system()" takes a while to complete won't affect the
 *   main reader at all.
 */
void
handle_pipe_input (int pfd[])
{
	FILE *fp;
	char buffer[512];
	
	/*
	 * Use stdio--more convenient
	 */
	fp = fdopen (pfd[0], "r");
	
	/*
	 * While there's still data available
	 */
	while (fgets (buffer, sizeof(buffer)-1, fp) != NULL)
	{
		char *p;
		
		/*
		 * Remove the trailing newline
		 */
		p = strchr (buffer, '\n');
		if (p)
		{
			*p = '\0';
		}
		
		/*
		 * And send it off to system()
		 */
		system (buffer);
	}
	fclose (fp);
}
