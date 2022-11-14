#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>

int  FLOWGRAPH_PORT = 13777;

int find_probable_pi_code (int infd, unsigned short *pi_code);
int process_rds_frame (int infd);
unsigned int getbits (int offs, int width, unsigned int value);
unsigned int calc_syndrome(unsigned long message, unsigned char mlen);
char *to_canadian_call(unsigned short picode);
void handle_groupcode_2A(unsigned int blocks[], unsigned int rdata);
void update_rds_gui (void);
void write_flowgraph_variable (char *pname, char *value, int port);
char cdn_table[0xCFFF-0xC000][5];
char *to_ptype[] =
{
        "No program type or undefined",
        "News",
        "Information",
        "Sports",
        "Talk",
        "Rock",
        "Classic rock",
        "Adult hits",
        "Soft rock",
        "Top 40",
        "Country",
        "Oldies",
        "Soft",
        "Nostalgia",
        "Jazz",
        "Classical",
        "Rhythm and blues",
        "Soft rhythm and blues",
        "Language",
        "Religious music",
        "Religious talk",
        "Personality",
        "Public",
        "College",
        "Unassigned",
        "Unassigned",
        "Unassigned",
        "Unassigned",
        "Unassigned",
        "Weather",
        "Emergency test",
        "Emergency"
};

int
main (int argc, char **argv)
{
	int infd;
	time_t now, then;
	int i;
	
	if ( argc < 3)
	{
		fprintf (stderr, "Usage: %s FIFO-file port\n", argv[0]);
		exit (0);
	}
	
	time (&now);
	time (&then);
	infd = open (argv[1], O_RDONLY);
	
	FLOWGRAPH_PORT = atoi(argv[2]);

	/*
	 * Pre-stuff for most Canadian stations
	 */
	for (i = 0xC000; i < 0xCFFF; i++)
	{
		strcpy (cdn_table[i-0xC000], to_canadian_call((unsigned short)i));
	}
	
	while (1)
	{
		if (process_rds_frame (infd) < 0)
		{
			break;
		}
		time (&now);
		if ((now - then) >= 2)
		{
			then = now;
			update_rds_gui ();
		}
	}
	
	exit(0);
}

/*
 * Frame length (bits)
 */
#define RDS_FRAME_LENGTH 104


#define UNSYNCED 0
#define SYNCED   1

static const unsigned int syndrome[5]={383,14,303,663,748};
static const unsigned int offset_word[5]={252,408,360,436,848};

/*
 * State variables we keep across invocations to process_rds_frame()
 */
int state = UNSYNCED;
char radiotext[65];
char ptype[32];
char station_call[32];
char last_update[32];
int badblock = 0;

int
process_rds_frame (int infd)
{
	unsigned int reg;
	unsigned short checkbits;
	unsigned char byte;
	unsigned char rds_bit_buffer[RDS_FRAME_LENGTH-16];
	int i, j;
	int bitcnt;
	unsigned int blocks[3];
	unsigned int dataword;
	unsigned int reg_syndrome;
	unsigned int blocknext;
	unsigned int bitcounter;
	unsigned short pi_code;
	
	reg = 0;
	blocknext = 0;
	bitcounter = 0;
	pi_code = 0;
	
	/*
	 * If we're not marked SYNCED, shift bits through, and try to get consecutive blocks
	 *   with matching CRCs.  Once you have that, you're SYNCed.  Theoretically.
	 */
	if (state != SYNCED)
	{
		for (;;)
		{
			reg <<= 1;
			if (read (infd, &byte, 1) != 1)
			{
				return -1;
			}
			bitcounter++;

			/*
			 * We've seen a lot of bits, reset our state a bit and start over
			 */
			if (bitcounter > (104*8))
			{
				blocknext = 0;
				bitcounter = 0;
			}
			reg = reg | (byte&0x01);
			dataword = (reg>>10)&0xFFFF;
			checkbits = reg&0x3FF;
			checkbits ^= offset_word[blocknext];
			reg_syndrome = calc_syndrome (dataword, 16);
			
			/*
			 * Once we have a matching CRC, bump our "blocknext"
			 * If we've seen 3 blocks, conclude SYNCed, update state
			 *   variable and return in new state from main()
			 * 
			 */
			if (reg_syndrome == checkbits)
			{
				if (blocknext >= 3)
				{
					state = SYNCED;
					return 0;
				}
				blocknext++;
			}
		}
	}
	
	/*
	 * If we're SYNCed, the first block we see *should* be an 'A' block with
	 *   a PI code in it.  Theoretically.
	 */
	else
	{
		bitcounter = 0;
		for (;;)
		{
			reg <<= 1;
			if (read (infd, &byte, 1) != 1)
			{
				return -1;
			}
			reg = reg | (byte&0x01);
			dataword = (reg>>10)&0xFFFF;
			checkbits = reg&0x3FF;
			checkbits ^= offset_word[0];
			reg_syndrome = calc_syndrome (dataword, 16);
			
			if (reg_syndrome == checkbits)
			{
				pi_code = dataword;
				break;
			}
		}
	}

	/*
	 * We got a PI code header, so read the next (RDS_FRAME_LENGTH-26) bits and stuff away
	 */
	for (i = 0; i < (RDS_FRAME_LENGTH-26); i++)
	{
		if (read (infd, &rds_bit_buffer[i], 1) != 1)
		{
			return -1;
		}
		rds_bit_buffer[i] &= 0x01;
	}
	
	/*
	 * Turn it into blocks (unsigned ints)
	 */
	bitcnt = 0;
	for (i = 0; i < 3; i++)
	{
		blocks[i] = 0;
		for (j = 0; j < 26; j++)
		{
			blocks[i] <<= 1;
			blocks[i] |= rds_bit_buffer[bitcnt];
			bitcnt++;
		}
	}

	/*
	 * Process the B block, and callout to C/D block processing
	 */
	{
		unsigned int groupcode;
		unsigned int b0;
		unsigned int tp;
		unsigned int pty;
		unsigned int rdata;
		unsigned int check;
		unsigned int calc;
		
		groupcode = getbits(22,4,blocks[0]);
		
		b0 = getbits(21,1,blocks[0]);
		
		tp = getbits(20,1,blocks[0]);
		
		pty = getbits(15,5,blocks[0]);
		
		rdata = getbits(10,5,blocks[0]);
		
		check = getbits(0,10,blocks[0]);
		check ^= offset_word[1];
		
		dataword = (blocks[0]>>10)&0xFFFF;
		calc = calc_syndrome (dataword, 16);
		
		/*
		 * B block OK
		 */
		if (check == calc)
		{
			
			/*
			 * North American mapping only for now
			 */
			strcpy (ptype, to_ptype[pty]);
			
			/*
			 * Handle Canadian call signs
			 * 
			 * Ideally, we want something that hanndles all of North America at least.
			 *   Later.  Handling Canada for now is fine with me :-)
			 */
			/*
			 * The great commercial-scum unwashed first
			 */
			if (pi_code >= 0xC000 && pi_code <= 0xCFFF)
			{
				strcpy (station_call, cdn_table[pi_code-0xC000]);
			}
			
			/*
			 * Her majesty's broadcasting corporation
			 * (And the USAn equivalent, I guess).
			 */
			else if ((pi_code & 0xF000) == 0xB000)
			{
				int x;
				char *network;
				
				network = "CBC";
				switch (pi_code & 0x000F)
				{
				case 1:
					network = "NPR";
					x = 1;
					break;
					
				case 2:
				case 4:
					x = 1;
					break;
					
				case 3:
				case 5:
					x = 2;
					break;
					
				default:
					x = 9;
					break;
				}
				sprintf (station_call, "%s-%d", network, x);
			}
			
			/*
			 * Rest of world is just raw PI code for now
			 */
			else
			{
				sprintf (station_call, "%04X", pi_code&0xFFFF);
			}
			
			/*
			 * Now, process C/D blocks
			 * For type 2A
			 */
			if (groupcode == 2 && b0 == 0)
			{
				struct tm *ltp;
				time_t now;
				
				time (&now);
				ltp = localtime(&now);
				sprintf (last_update, "%02d:%02d:%02d",
					ltp->tm_hour,
					ltp->tm_min,
					ltp->tm_sec);
					
				handle_groupcode_2A (blocks, rdata);
			}
			badblock = 0;
		}
		
		/*
		 * Kind of abitrary, but it seems to provide fast resynch after frequency change, etc
		 */
		else if (badblock++ > 5)
		{
			strcpy (station_call, "????");
			strcpy (ptype, "????");
			memset (radiotext, ' ', sizeof(radiotext)-1);
			state = UNSYNCED;
			
			/*
			 * Tell the flow-graph to reset the modulator, just in case that's what's causing us
			 *    to get a lot of bad blocks.
			 */
			write_flowgraph_variable ("mod_reset", "1", FLOWGRAPH_PORT);
			badblock = 0;
		}
		
	}
	return (0);
}

int abflag;

void
handle_groupcode_2A (unsigned int blocks[], unsigned int rdata)
{
	unsigned int dataword;
	unsigned int checkbits;
	unsigned int calc;
	int i;
	int ndx;
	
	ndx = rdata&0xF;
	
	/*
	 * Standard says to dump existing text when A/B flag flips
	 */
	if (abflag != (rdata&0x10))
	{
		abflag = rdata&0x10;
		memset (radiotext, ' ', sizeof(radiotext)-1);
	}
	
	/*
	 * For C/D words
	 */
	for (i = 1; i <= 2; i++)
	{
		unsigned short datashort;
		
		dataword = (blocks[i]>>10)&0xFFFF;
		datashort = dataword & 0xFFFF;
		
		checkbits = (blocks[i]&0x3FF);
		checkbits ^= offset_word[i+1];
		calc = calc_syndrome (dataword, 16);
		
		/*
		 * CRC passed
		 */
		if (calc == checkbits)
		{
			int rndx;		
			unsigned char tmp;
			unsigned char cbuf[2];
			char *p2;
			
			/*
			 * Looks like on X86 we have to exchange bytes 0/1
			 */
			memcpy (cbuf, &datashort, sizeof(datashort));
			tmp = cbuf[1];
			cbuf[1] = cbuf[0];
			cbuf[0] = tmp;
			
			
			rndx = (i-1)*2;
			
			/*
			 * Update radiotext
			 */
			memcpy (&radiotext[(ndx*4)+rndx], cbuf, 2);
			
			/*
			 * Convert various troublesome characters
			 */
			if ((p2 = strchr (radiotext, '\r')) )
			{
				*p2 = ' ';
			}
			if ((p2 = strchr (radiotext, '\n')) )
			{
				*p2 = ' ';
			}
			if ((p2 = strchr (radiotext, '\t')) )
			{
				*p2 = ' ';
			}
			if ((p2 = strchr (radiotext, '\'')) )
			{
				*p2 = '_';
			}
			if ((p2 = strchr (radiotext, '"')) )
			{
				*p2 = '_';
			}
		}
	}
}
unsigned int
getbits (int offs, int width, unsigned int value)
{
	unsigned int msk;
	unsigned int val;
	
	msk = (1<<width)-1;
	msk <<= (offs);
	
	val = value&msk;
	val >>= offs;
	
	return (val);
}

/* see Annex B, page 64 of the standard */
unsigned int calc_syndrome(unsigned long message, unsigned char mlen)
{
	unsigned long reg=0;
	unsigned int i;
	const unsigned long poly=0x5B9;
	const unsigned char plen=10;

	for (i=mlen;i>0;i--)
	{
		reg=(reg<<1) | ((message>>(i-1)) & 0x01);
		if (reg & (1<<plen)) reg=reg^poly;
	}
	for (i=plen;i>0;i--)
	{
		reg=reg<<1;
		if (reg & (1<<plen)) reg=reg^poly;
	}
	return (reg & ((1<<plen)-1));	// select the bottom plen bits of reg
}

/*
 * Code adapted from C++ routine supplied by Tom Pittenger--tpittenger@ieee.org
 */
char *
to_canadian_call (unsigned short pi_code)
{
	static char retbuf[32];
	char *p;
	int i, j, k;
	int G, H;
	
	if ((pi_code & 0xF000) == 0xC000)
	{
		int PI_data = pi_code;
		
		p = retbuf;
        PI_data -= 49152;
		for (i = 0; i < 5; i++)
		{
			for (j = 0; j < 26; j++)
			{
				for (k = 0; k < 27; k++)
				{
					// G = i * 676 + j * 27 + k + 257; // Rev 1
					G = i * 702 + j * 27 + k + 257; // Rev 2
					H = (G - 257) / 255;
					if ((G + H) == PI_data)
					{
						*p++ = 'C';
						if (i == 0)
						{
							*p++ = 'F'; 
						}
						else
						{
							*p++ = i+'G';
						}
						*p++ = j + 'A';
						if (k == 0) 
						{
							*p++ = ' ';
						}
						else
						{
							*p++ = k + 64;
						}
						*p++ = '\0';
						return (retbuf);
					} // G+H
				} // for k
			} // for j
		} // for i
	}
	return (retbuf);
}

void
update_rds_gui ()
{
	write_flowgraph_variable ("ptype", ptype, FLOWGRAPH_PORT);
	write_flowgraph_variable ("stname", station_call, FLOWGRAPH_PORT);
	write_flowgraph_variable ("rtext", radiotext, FLOWGRAPH_PORT);
	write_flowgraph_variable ("lastupdate", last_update, FLOWGRAPH_PORT);
}

/*
 * Update a variable within the flow-graph
 */
void
write_flowgraph_variable (char *pname, char *value, int port)
{
	char sys_str[512];
	
	sprintf (sys_str, 
		"python -c \"import xmlrpclib;xmlrpclib.Server('http://localhost:%d').set_%s('%s')\"\n",
		port, pname, value);
		
	system (sys_str);
}
