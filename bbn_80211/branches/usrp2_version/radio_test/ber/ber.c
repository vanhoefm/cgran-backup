#include <errno.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netpacket/packet.h>
#include <net/ethernet.h>
#include <netinet/in.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <signal.h>

/* 
 * This lookup table is used for counting bit errors.
 * bitErrorTable[i] gives the number of bits equal to 1 in the binary
 * representation of i. 
 */
static const int bitErrorTable[256] = {
  0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 
  1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 
  1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 
  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 
  1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 
  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 
  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 
  3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 
  1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 
  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 
  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 
  3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 
  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 
  3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 
  3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 
  4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};

static const unsigned long packetHeader[4] = {
  0x00010203, 0x04050001, 0x02030406, 0x87640000
};


#define DEFAULT_NUM_PACKETS 10000
#define DEFAULT_BIT_RATE 500000
#define DEFAULT_PACKET_LEN 100
#define DEFAULT_DIFS 0
#define DEFAULT_REPORT_STATS_INTERVAL 10
#define DEFAULT_IFNAME "gr0"
#define DEFAULT_SHOW_PKT_BYTES 0

typedef enum menuModeEnum {
  menuModeMain,
  menuModeConfig
} menuMode_t;

typedef enum mainMenuCommandEnum {
  mainMenuSendTestPacket = 1,
  mainMenuSendManyPackets,
  mainMenuShowStats,
  mainMenuResetStats,
  mainMenuChangeConfig,
  mainMenuQuit
} mainMenuCommand_t;

typedef enum configMenuCommandEnum {
  configMenuChooseItem,
  configMenuNumPackets,
  configMenuBitRate,
  configMenuPacketLength,
  configMenuDifs,
  configMenuStatsInterval,
  configMenuShowPackets,
  configMenuInterface,
  configMenuLastOption
} configMenuCommand_t;

typedef struct berConfigStruct {
  int numPackets; /* Number of packets to send */
  unsigned long bitRate;   /* bits/sec */
  int packetLen; /* bytes */
  int difs;      /* microseconds between sent packets */
  int reportStatsInterval; /* Number of packets before reporting the stats. */
  int showPktBytes; /* When true, packet contents are displayed */
  char ifName[IFNAMSIZ + 1];
} berConfig_t;

typedef struct berStatsStruct {
  int rcvPacketCount;
  int pktErrors;
  int rcvBitCount;
  int bitErrors;
  int wrongSizedPackets;
} berStats_t;

static int openRawDevice(char *ifName, unsigned short pktType);
static void dumpPacket(void *data, int len);
static void sendTestPacket(int sd);
static int readPacket(int sd, char *buf, int bufLen, int *packetType);
static int handleBerPacket(int sd);
static int handleStdIn(int sd);
static void intHandler(int signalIn);
static void displayMainMenu(void);
static void displayConfigMenu(configMenuCommand_t mode, berConfig_t *cfg);
static void initConfig(berConfig_t *cfg);
static void changeConfig(berConfig_t *cfg);
static void initStats(berStats_t *stats);
static void showStats(berStats_t *stats);
static int sendManyPackets(int sd);
static int analyzePacket(void *packetData, int len);
static void usage(void);
static int parseArgs(int argc, char **argv, berConfig_t *cfg);
static inline int numOnes(unsigned long word);

#define BUFLEN 1500

static int packetSD;
static int timeToExit = 0;
static menuMode_t menuMode;
static configMenuCommand_t configMenuCommand;
static berConfig_t config;
static berStats_t stats;

static void usage(void) {
  fprintf(stderr, "Usage: ber [options]\n");
  fprintf(stderr, "   -num_packets    <Number of packets to send>\n");
  fprintf(stderr, "   -bit_rate       <bit rate in bits per second>\n");
  fprintf(stderr, "   -difs           <microseconds between sent packets>\n");
  fprintf(stderr, "   -stats_interval <number of receive packets before "
	  "reporting statistics>\n");
  fprintf(stderr, "   -show_packets   <Show received packet contents>\n");
  fprintf(stderr, "   -if_name        <interface name>\n\n");
}

static int parseArgs(int argc, char **argv, berConfig_t *cfg) {
  int curArg;
  int iVal;
  double dVal;

  for(curArg = 1; curArg < argc; ++curArg) {
    if(!strcmp(argv[curArg], "-help")) {
      usage();
      exit(0);
    } else if(!strcmp(argv[curArg], "--help")) {
      usage();
      exit(0);
    } else if(!strcmp(argv[curArg], "-num_packets")) {
      ++curArg;
      if(curArg == argc) {
	usage();
	return -1;
      }
      iVal = atoi(argv[curArg]);
      if(iVal < 1) {
	fprintf(stderr, "ber: Number of packets must be greater than zero.\n");
	usage();
	return -1;
      }
      cfg->numPackets = iVal;
    } else if(!strcmp(argv[curArg], "-bit_rate")) {
      ++curArg;
      if(curArg == argc) {
	usage();
	return -1;
      }
      if(sscanf(argv[curArg], "%lf", &dVal) == 0) {
	usage();
	return -1;
      }

      if(dVal <= 1) {
	fprintf(stderr, "ber: Bit rate must be greater than one.\n");
	usage();
	return -1;
      }
      cfg->bitRate = (int)dVal;
     } else if(!strcmp(argv[curArg], "-difs")) {
      ++curArg;
      if(curArg == argc) {
	usage();
	return -1;
      }
      if(sscanf(argv[curArg], "%lf", &dVal) == 0) {
	usage();
	return -1;
      }

      if(dVal <= 1) {
	fprintf(stderr, "ber: Bit rate must be greater than one.\n");
	usage();
	return -1;
      }
      cfg->difs = (int)dVal;
    } else if(!strcmp(argv[curArg], "-stats_interval")) {
      ++curArg;
      if(curArg == argc) {
	usage();
	return -1;
      }
      iVal = atoi(argv[curArg]);
      if(iVal < 0) {
	fprintf(stderr, "ber: Statistics interval must be non-negative."
		"\n");
	usage();
	return -1;
      }
      cfg->reportStatsInterval = iVal;
    } else if(!strcmp(argv[curArg], "-show_packets")) {
      cfg->showPktBytes = 1;
    } else if(!strcmp(argv[curArg], "-if_name")) {
      ++curArg;
      if(curArg == argc) {
	usage();
	return -1;
      }
      strncpy(cfg->ifName, argv[curArg], IFNAMSIZ);
    } else {
      fprintf(stderr, "ber: Unknown option \"%s\"\n\n", argv[curArg]);
      usage();
      return -1;
    }
  }

  return 0;
}

static void intHandler(int signalIn) {
  timeToExit = 1;
  signal(signalIn, intHandler);
}

static void initConfig(berConfig_t *cfg) {
  cfg->numPackets = DEFAULT_NUM_PACKETS;
  cfg->bitRate = DEFAULT_BIT_RATE;
  cfg->packetLen = DEFAULT_PACKET_LEN;
  cfg->difs = DEFAULT_DIFS;
  cfg->reportStatsInterval = DEFAULT_REPORT_STATS_INTERVAL;
  cfg->showPktBytes = DEFAULT_SHOW_PKT_BYTES;
  strncpy(cfg->ifName, DEFAULT_IFNAME, IFNAMSIZ);
  cfg->ifName[IFNAMSIZ] = '\0';
}

static void initStats(berStats_t *stats) {
  stats->rcvPacketCount = 0;
  stats->pktErrors = 0;
  stats->rcvBitCount = 0;
  stats->bitErrors = 0;
  stats->wrongSizedPackets = 0;
}

static void displayConfigMenu(configMenuCommand_t mode, berConfig_t *cfg) {
  switch(mode) {
  case configMenuChooseItem:
    printf("\n");
    printf("---- Configuration Menu ----\n\n");
    printf("1.  Number of packets    : %d\n", cfg->numPackets);
    printf("2.  Bit rate             : %.4g\n", (double)cfg->bitRate);
    printf("3.  Packet length        : %d\n", cfg->packetLen);
    printf("4.  Time between packets : %d\n", cfg->difs);
    if(cfg->reportStatsInterval == 0) {
      printf("5.  Statistics Interval  : OFF\n");
    } else {
      printf("5.  Statistics Interval  : %d\n", cfg->reportStatsInterval);
    }
    if(cfg->showPktBytes) {
      printf("6.  Show Packet Contents : YES\n");
    } else { 
      printf("6.  Show Packet Contents : NO\n");
   }
    printf("7.  Interface            : %s\n\n", cfg->ifName);
    printf("   Select Parameter to change ==> ");
    break;
  case configMenuNumPackets:
    printf("     Enter Number of packets ==> ");
    break;
  case configMenuPacketLength:
    printf("     Enter Packet Length in bytes ==> ");
    break;
  case configMenuBitRate:
    printf("     Enter Bit Rate ==> ");
    break;
  case configMenuDifs:
    printf("     Enter timer between packets in microseconds ==> ");
    break;
  case configMenuStatsInterval:
    printf("     Enter Statistics Interval ==> ");
    break;
  case configMenuShowPackets:
    break;
  case configMenuInterface:
    printf("     Enter interface name ==> ");
    break;
  default:
    break;
  }

  fflush(stdout);
}

static void changeConfig(berConfig_t *cfg) {
  menuMode = menuModeConfig;
  configMenuCommand = configMenuChooseItem;
  displayConfigMenu(configMenuCommand, cfg);
}

static void showStats(berStats_t *stats) {
  double BER;
  double PER;

  if(stats->rcvPacketCount == 0) {
    PER = 0;
  } else {
    PER = (double)stats->pktErrors / 
      (double)stats->rcvPacketCount;
  }

  if(stats->rcvBitCount == 0) {
    BER = 0;
  } else {
    BER = (double)stats->bitErrors /
      (double)stats->rcvBitCount;
  }

  printf("\n\n");
  printf("  Receive Packets        : %d\n", stats->rcvPacketCount);
  printf("  Received Packet Errors : %d\n", stats->pktErrors);
  printf("  Packet Error Rate      : %.4lg\n", PER);
  printf("  Recieved Bits          : %d\n", stats->rcvBitCount);
  printf("  Receive Bit Errors     : %d\n", stats->bitErrors);
  printf("  Bit Error Rate         : %.4lg\n", BER);
  printf("  Wrong Sized Packets    : %d\n", stats->wrongSizedPackets);
}

static void dumpPacket(void *data, int len) {
  unsigned short *dp;
  int lineLen;
  int i;
 
  dp = (unsigned short *)data;

  while( len > 0 ) {
    if(len > 16) {
      lineLen = 16;
    } else {
      lineLen = len;
    }

    for(i=0; i < (lineLen >> 1); ++i) {
      printf("0x%04x ", ntohs(dp[0]));
      ++dp;
    }
    if(lineLen & 0x01) {
      printf("0x%02x ", ntohs(dp[0] & 0x00ff));
    }
    len -= lineLen;
    printf("\n");
  }

  printf("\n");
}

static void sendTestPacket(int sd) {
  char testBuffer[1600];
  unsigned long *lData;
  int i;

  lData = (unsigned long *)testBuffer;

  bzero(testBuffer, sizeof(testBuffer));
  lData[0] = ntohl(0x00010203);
  lData[1] = ntohl(0x04050001);
  lData[2] = ntohl(0x02030406);
  lData[3] = ntohl(0x8764aaaa);
  lData[4] = ntohl(0x5555aaaa);
  lData[5] = ntohl(0x5555aaaa);
  lData[6] = ntohl(0x5555aaaa);
  lData[7] = ntohl(0x5555aaaa);

  write(sd, testBuffer, 32);
}

static int sendManyPackets(int sd) {
  char packetBuffer[1600];
  int pktLen;
  unsigned long *lData;
  int i;
  long delayUS;
  int bitsPerMS;
  double startTime, endTime;
  struct timeval now;
  double pktsPerSec, bytesPerSec, bitsPerSec;

  pktLen = config.packetLen;
  if(pktLen > 1600) {
    pktLen = 1600;
  } else if(pktLen < 14) {
    pktLen = 14;
  }

  lData = (unsigned long *)packetBuffer;

  bzero(packetBuffer, pktLen);
  for(i=0; i<3; ++i) {
    lData[i] = ntohl(packetHeader[i]);
  }

  srand(13);
  for(i=0; i < ((pktLen >> 2) - 4); ++i) {
    lData[i + 4] = htonl(rand());
  }

  bitsPerMS = config.bitRate / 1000;
  delayUS = ((pktLen * 8000) / bitsPerMS) + config.difs;
  
  gettimeofday(&now, NULL);
  startTime = (double)now.tv_sec + ( (double)now.tv_usec) / 1000000.0;

  for(i=0; i < config.numPackets; ++i) {
    /* Add sequence Number */
    lData[3] = ntohs(packetHeader[i] | i);

    if(write(sd, packetBuffer, pktLen) != pktLen) {
      perror("ber: write");
      return -1;
    }
    usleep(delayUS);
    if(timeToExit) {
      ++i;
      timeToExit = 0;
      break;
    }
  }

  gettimeofday(&now, NULL);
  endTime = (double)now.tv_sec + ( (double)now.tv_usec) / 1000000.0;

  /* This just flushes the tap buffers, not the gnu radio buffers. */
  fsync(sd);
 
  printf("\nTransmitted %d Packets int %.3g seconds.\n", 
	 i, endTime - startTime);
  if((endTime - startTime) > 0) {
    pktsPerSec = (double)i / (endTime - startTime);
    bytesPerSec = (double)pktLen * pktsPerSec;
    bitsPerSec = bytesPerSec * 8.0;

    printf("Average Rate = %.3g packets/sec, %.3f bytes/sec, %.3f bits/sec.\n",
	   pktsPerSec, bytesPerSec, bitsPerSec);
  }

  return 0;
}

static int readPacket(int sd, char *buf, int bufLen, int *packetType) {
  int result;
  struct sockaddr_ll sockaddr;
  socklen_t fromLen;

  fromLen = sizeof(sockaddr);
  bzero(&sockaddr, sizeof(sockaddr));
  sockaddr.sll_family = AF_PACKET;
  sockaddr.sll_protocol = htons(ETH_P_ALL);
  result = recvfrom(sd, buf, bufLen, 0,
		    (struct sockaddr *)&sockaddr, &fromLen);
  if(result < 0) {
    perror("ber: recvfrom");
    return -1;
  }

  *packetType = sockaddr.sll_pkttype;

  return result;
}

static inline int numOnes(unsigned long word) {
  int result;

  result = 0;
  result += bitErrorTable[word & 0xff];
  word >>= 8;
  result += bitErrorTable[word & 0xff];
  word >>= 8;
  result += bitErrorTable[word & 0xff];
  word >>= 8;
  result += bitErrorTable[word & 0xff];

  return result;
}

static int analyzePacket(void *packetData, int len) {
  unsigned int aLen;
  unsigned long *lData;
  unsigned long nextDataWord;
  unsigned long diff;
  int i;
  int packetError;

  ++stats.rcvPacketCount;
  if(len != config.packetLen) {
    ++stats.wrongSizedPackets;
    if(len > config.packetLen) {
      aLen = config.packetLen;
    } else {
      aLen = len;
    }
  } else {
    aLen = len;
  }

  stats.rcvBitCount += aLen * 8;

  packetError = 0;
  lData = (unsigned long*)packetData;

  for(i=0; i<4; ++i) {
    nextDataWord = packetHeader[i];
    diff = htonl(*lData) ^ nextDataWord;

    if(diff) {
      packetError = 1;
      stats.bitErrors += numOnes(diff);
    }

    ++lData;
    aLen -= 4;
  }

  srand(13);
  while(aLen > 0) {
    nextDataWord = rand();
    diff = htonl(*lData) ^ nextDataWord;
    if(aLen < 4) {
      switch(aLen) {
      case 1:
	diff &= 0xffffff00;
	break;
      case 2:
	diff &= 0xffff0000;
	break;
      case 3:
	diff &= 0xff000000;
	break;
      }
    }

    if(diff) {
      packetError = 1;
      stats.bitErrors += numOnes(diff);
    }

    ++lData;
    aLen -= 4;
  }
  
  if(packetError != 0) {
    ++stats.pktErrors;
  }

  return 0;
}

static int openRawDevice(char *ifName, unsigned short pktType) {
  int sd;
  int result;
  struct sockaddr_ll sockaddr;
  struct ifreq ifr;
  
  strncpy(ifr.ifr_name, ifName, IFNAMSIZ);
  ifr.ifr_name[IFNAMSIZ - 1] = '\0';

  printf("Opening device %s\n", ifr.ifr_name);

  sd = socket(PF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
  if(sd < 0) {
    perror("ber: socket");
    return sd;
  }

  result = ioctl(sd, SIOGIFINDEX, &ifr);
  if(result == -1) {
    close(sd);
    fprintf(stderr, "ber: Unable to connect to \"%s\".\n", 
	    ifr.ifr_name);
    perror("ber: ioctl");
    return -1;
  }

  sockaddr.sll_family = AF_PACKET;
  sockaddr.sll_protocol = htons(ETH_P_ALL);
  sockaddr.sll_protocol = htons(pktType);
  sockaddr.sll_ifindex = ifr.ifr_ifindex;

  result = bind(sd, (struct sockaddr *)&sockaddr, 
		sizeof(struct sockaddr_ll));
  if(result == -1) {
    close(sd);
    perror("ber: bind");
    return -1;
  }

  return sd;
}

static int handleBerPacket(int sd) {
  int result;  
  char buf[BUFLEN];
  int pktType;

  result = readPacket(sd, buf, BUFLEN, &pktType);

  if(result <= 0) {
    return result;
  }

  if(pktType == PACKET_OUTGOING) {
    return 0;
  }

  if(config.showPktBytes) {
    dumpPacket(buf, result);
  }

  analyzePacket(buf, result);

  if(config.reportStatsInterval > 0) {
    if((stats.rcvPacketCount % config.reportStatsInterval) == 0) {
      showStats(&stats);
    }
  }

  return 0;
}

static int handleStdIn(int sd) {
  int result;
  char buf[BUFLEN];
  char buf2[BUFLEN];
  int selection;
  double dselection;
  int newSd;

  result = read(0, buf, BUFLEN - 1);

  if(result <= 0) {
    return result;
  }

  buf[result] = '\0';

  fflush(stdout);

  switch(menuMode) {
  case menuModeMain:
    if(sscanf(buf, "%d %s", &selection, buf2) == 1) {
      switch(selection) {
      case mainMenuSendTestPacket:
	sendTestPacket(sd);
	displayMainMenu();
	break;
      case mainMenuSendManyPackets:
	sendManyPackets(sd);
	displayMainMenu();
	break;
      case mainMenuShowStats:
	showStats(&stats);
	displayMainMenu();
	break;
      case mainMenuResetStats:
	initStats(&stats);
	displayMainMenu();
	break;
      case mainMenuQuit:
	timeToExit = 1;
	break;
      case mainMenuChangeConfig:
	changeConfig(&config);
	break;
      default:
	displayMainMenu();
	break;
      }
    } else {
      displayMainMenu();
    }
    break;
  case menuModeConfig:
    buf2[0] = '\0';
    sscanf(buf, "%s", buf2);
    buf2[BUFLEN - 1] = '\0';

    switch(configMenuCommand) {
    case configMenuChooseItem:
      if(buf2[0] == '\0') {
	menuMode = menuModeMain;
	displayMainMenu();
      } else {
	selection = atoi(buf2);
	if((selection > 0) && (selection < configMenuLastOption)) {
	  if(selection == configMenuShowPackets) {
	    config.showPktBytes = !config.showPktBytes;
	  } else {
	    configMenuCommand = selection;
	  }
	} else {
	  fprintf(stderr, "ERROR: Please Enter a number between 1 and %d.\n",
		  configMenuLastOption);
	}
	displayConfigMenu(configMenuCommand, &config);
      }
      break;
    case configMenuNumPackets:
      if(buf2[0] != '\0') {
	selection = atoi(buf2);
	if(selection > 0) {
	  config.numPackets = selection;
	} else {
	  fprintf(stderr, "ERROR: Please enter a positive integer.\n");
	}
      }
      configMenuCommand = configMenuChooseItem;
      displayConfigMenu(configMenuCommand, &config);
      break;
    case configMenuBitRate:
      if(sscanf(buf2, "%lf", &dselection) == 1) {
	selection = (int)dselection;
	if(selection > 0) {
	  config.bitRate = selection;
	} else {
	  fprintf(stderr, "ERROR: Please enter a positive integer.\n");
	}
      }
      configMenuCommand = configMenuChooseItem;
      displayConfigMenu(configMenuCommand, &config);
      break;
    case configMenuPacketLength:
      if(buf2[0] != '\0') {
	selection = atoi(buf2);
	if(selection > 0) {
	  config.packetLen = selection;
	} else {
	  fprintf(stderr, "ERROR: Please enter a positive integer.\n");
	}
      }
      configMenuCommand = configMenuChooseItem;
      displayConfigMenu(configMenuCommand, &config);
      break;
    case configMenuDifs:
      if(sscanf(buf2, "%lf", &dselection) == 1) {
	selection = (int)dselection;
	if(selection >= 0) {
	  config.difs = selection;
	} else {
	  fprintf(stderr, "ERROR: Please enter a non-negative integer.\n");
	}
      }
      configMenuCommand = configMenuChooseItem;
      displayConfigMenu(configMenuCommand, &config);
      break;
    case configMenuStatsInterval:
      if(buf2[0] != '\0') {
	selection = atoi(buf2);
	if(selection >= 0) {
	  config.reportStatsInterval = selection;
	} else {
	  fprintf(stderr, "ERROR: Please enter a positive integer.\n");
	}
      }
      configMenuCommand = configMenuChooseItem;
      displayConfigMenu(configMenuCommand, &config);
      break;
    case configMenuShowPackets:
      break;
    case configMenuInterface:
      if(buf2[0] != '\0') {
	if(strcmp(buf2, config.ifName)) {
	  newSd = openRawDevice(buf2, 0x8764);
	  if(newSd < 0) {
	    break;
	  }
	  printf("newSd = %d\n", newSd);
	  close(packetSD);
	  strncpy(config.ifName, buf2, IFNAMSIZ);
	  packetSD = newSd;
	  sd = packetSD;
	}
      }
      configMenuCommand = configMenuChooseItem;
      displayConfigMenu(configMenuCommand, &config);
      break;
    }
  }

  return 0;
}

int main(int argc, char **argv) {
  int result;
  fd_set readfds;

  initConfig(&config);

  if(parseArgs(argc, argv, &config) < 0) {
    return 255;
  }

  packetSD = openRawDevice(config.ifName, 0x8764);
  if(packetSD < 0) {
    return 255;
  }

  initStats(&stats);
  menuMode = menuModeMain;

  displayMainMenu();
  signal(SIGINT, intHandler);
  while(timeToExit == 0) {
    FD_ZERO(&readfds);
    FD_SET(packetSD, &readfds);
    FD_SET(0, &readfds); /* Standard Input */

    result = select(packetSD + 1, &readfds, NULL, NULL, NULL);
    if(timeToExit) {
      break;
    }

    if(result < 0) {
      perror("ber: select");
      close(packetSD);
      return -1;
    }

    if(FD_ISSET(packetSD, &readfds)) {
      result = handleBerPacket(packetSD);
      if(result < 0) {
	return -1;
      }      
    }

    if(FD_ISSET(0, &readfds)) {
      result = handleStdIn(packetSD);
      if(result < 0) {
	return -1;
      }
    }

  }

  close(packetSD);
  printf("\n\n");
  fflush(stdout);
  return 0;
}

static void displayMainMenu(void) {
  printf("\n");
  printf("---- Main Menu ----\n\n");
  printf(" 1.   Send Test Packet\n");
  printf(" 2.   Send Many Packets\n");
  printf(" 3.   Show Statistics\n");
  printf(" 4.   Reset Statistics\n");
  printf(" 5.   Change a configuration parameter\n");
  printf(" 6.   Quit\n\n");
  printf("   Selection ===> ");
  fflush(stdout);
}
