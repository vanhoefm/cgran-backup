#include <stdio.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <stdlib.h>
#include <strings.h>

#define FLAG         0x7E
#define FRAME_MAX    8192
#define BIT_BUF_MAX 78644
#define HUNT            0
#define IDLE            1
#define FRAMING         2
#define SUCCESS         1
#define FAIL            0



//-------------------------

/*
 *                                      16   12   5
 * this is the CCITT CRC 16 polynomial X  + X  + X  + 1.
 * This is 0x1021 when x is 2, but the way the algorithm works
 * we use 0x8408 (the reverse of the bit pattern).  The high
 * bit is always assumed to be set, thus we only use 16 bits to
 * represent the 17 bit value.
 *
 * We have to use the reverse pattern because serial comm using
 * UARTs abd USRTs send out each byte LSBit first, but the
 * CRC algorithm is specified using MSBit first.  So this
 * whole algorithm is reversed to compensate.  The main loop
 * shifts right instead of left as in the cannonical algorithm
*/

#define POLY 0x8408   /* 1021H bit reversed */

unsigned short crc16(unsigned char *data_p, unsigned short length)
{
      unsigned char i;
      unsigned int data;
      unsigned int crc = 0xffff;

      if (length == 0)
            return (~crc);
      do
      {
            for (i=0, data=(unsigned int)0xff & *data_p++;
                 i < 8; 
                 i++, data >>= 1)
            {
                  if ((crc & 0x0001) ^ (data & 0x0001))
                        crc = (crc >> 1) ^ POLY;
                  else  crc >>= 1;
            }
      } while (--length);

      crc = ~crc;
      data = crc;
      crc = (crc << 8) | (data >> 8 & 0xff);

      return (crc);
}

//-------------------------


int
crc_valid(int frame_size, unsigned char * frame)
  {
    unsigned short frame_crc;
    unsigned short calc_crc;

    frame_crc = frame[frame_size-1] | (frame[frame_size-2] << 8);
    calc_crc = crc16(frame, frame_size-2);
    //printf("Frame_crc = %04X   Calc_crc = %04X\n", frame_crc, calc_crc);
    return(calc_crc == frame_crc);
  }        

void
decode_frame(int frame_size, unsigned char * frame);

int
unstuff(int             bit_buf_size, 
        unsigned char * bit_buf, 
        int *           frame_buf_size, 
        unsigned char * frame_buf)
  {
    int           i;
    unsigned char data_bit;
    int           accumulated_bits;
    int           bytes;
    int           consecutive_one_bits;
    int           status;
 
    accumulated_bits = 0;
    bytes = 0;
    consecutive_one_bits = 0;

    for(i=0; i<bit_buf_size; i++)
      {
        data_bit = bit_buf[i];
        if( (consecutive_one_bits != 5) || (data_bit == 1) )
          { 
            // Not a stuffed 0,  Write it to frame_buf
            frame_buf[bytes] = (frame_buf[bytes] >> 1) | (data_bit << 7);
            accumulated_bits++;
            if(accumulated_bits == 8)
              {
                bytes++;
                accumulated_bits = 0;
              }
          }

        if(data_bit == 1)
          {
            consecutive_one_bits++;
          }
        else
          {
            consecutive_one_bits = 0;
          }
      }

    // Now check for an integer number of bytes
    if(accumulated_bits == 0)
      {
        status = SUCCESS;
        *frame_buf_size = bytes;
      }
    else
      {
        printf("FAIL: frame_buf size = %d   accumulated_bits = %d\n", bytes, accumulated_bits);
        decode_frame(bytes, frame_buf);
        status = FAIL;
        *frame_buf_size = 0;
        //*frame_buf_size = bytes;
      }

    return status;
  }



int
get_bit(unsigned char* next_bit)
  {
#ifdef JUNK
    static int           bit_cnt = 0;  //Num of bits before we have to read another byte
    static unsigned char byte;
    int                  status;
    unsigned char        bit;

    //printf("bit_cnt = %d,  byte = %02X\n", bit_cnt, byte);

    if((bit_cnt == 0) && feof(stdin))
      {
        status = EOF;
      }
    else
      {
        if(bit_cnt == 0)
          {
            fread(&byte, sizeof(unsigned char), 1, stdin);
            bit_cnt = 8;
            //printf("Read another byte: %02X\n", byte);
          }
        // HDLC temporal bit order is lsb transmitted first
        *next_bit = byte & 0x01;
        byte = byte >> 1;
        bit_cnt--;
        status = 0;
      }
    return status;
#endif

    // Read unpacked bitstream from file
    int                  status;

    if(feof(stdin))
      {
        status = EOF;
      }
    else
      {
        fread(next_bit, sizeof(unsigned char), 1, stdin);
        status = 0;
      }
    return status;
    
  }


void
print_frame(int frame_size, unsigned char * frame)
  {
    int   i;

    for(i=0; i<frame_size; i++)
      {
        printf("%02X ", frame[i]);
      }
    printf("\n");
  }



void
decode_frame(int frame_size, unsigned char * frame)
  {
    print_frame(frame_size, frame);;

    if(frame[4] == 0x45)
      {
        printf("    IP Packet:\n");
        printf("        Size     : %d\n", (frame[6]*256+frame[7]));
        printf("        Src Addr : %d.%d.%d.%d\n", frame[16], frame[17], frame[18], frame[19]);
        printf("        Dst Addr : %d.%d.%d.%d\n", frame[20], frame[21], frame[22], frame[23]);
        if(frame[13] == 0x11)
          {
            printf("        Payload  : UDP Packet\n");
            printf("            Src Port  : %d\n", frame[24]*256+frame[25]);
            printf("            Dst Port  : %d\n", frame[26]*256+frame[27]);
            printf("            Size      : %d\n", frame[28]*256+frame[29]);
          }
      }
    printf("\n");
  }



void
route_packet(int hdlc_frame_size, unsigned char * hdlc_frame)
  {
    static int                ip_socket = 0;
    static struct sockaddr_in dest_addr;
    int                       packet_size;
    unsigned char *           packet;
    int                       flags;
    int                       hincl;
    int                       stat;

    if(ip_socket == 0)
      {
        bzero((char *) &dest_addr, sizeof(dest_addr));
        dest_addr.sin_family      = AF_INET;
        dest_addr.sin_port        = htons(0);
        dest_addr.sin_addr.s_addr = inet_addr("192.168.3.10");

        ip_socket = socket(AF_INET, SOCK_RAW, IPPROTO_RAW);
        if (ip_socket <=0)
          {
            fprintf(stderr, "Failure opening raw IP socket. Are you root?\n");
            perror("");
            exit(-1);
          }
        int hincl = 1;           /* 1 = on, 0 = off */
        stat = setsockopt(ip_socket, IPPROTO_IP, IP_HDRINCL, &hincl, sizeof(hincl));
        if(stat < 0)
          {
            fprintf(stderr, "setsockopt failed to set IP_HDRINCL.\n");
            exit(-1);
          }
      }

    packet_size = (hdlc_frame[6] << 8) | hdlc_frame[7];
    packet = hdlc_frame + 4;
    flags = 0;

    stat = sendto(ip_socket, 
                  packet, 
                  packet_size, 
                  flags, 
                  (struct sockaddr *)&dest_addr, 
                  sizeof(dest_addr));
    if(stat < 0)
      {
        perror("sendto failed");
        exit(-1);
      }
  }



int
main(int argc, char** argv)
  {
    int           state;
    int           next_state;
    unsigned char next_bit;
    unsigned char byte;
    int           flag_cnt;
    int           good_frame_cnt;
    int           good_byte_cnt;
    int           crc_err_cnt;
    int           abort_cnt;
    int           seven_ones_cnt;
    int           non_align_cnt;
    int           giant_cnt;
    int           runt_cnt;
    int           consecutive_one_bits;
    int           accumulated_bits;
    unsigned char bit_buf[BIT_BUF_MAX];
    int           bit_buf_size;
    unsigned char frame_buf[FRAME_MAX];
    int           frame_size;
    int           status;
    int           i;

    // Initialize state info
    state = HUNT;
    byte = 0x00;
    accumulated_bits = 0;

    // Initialize data statistics
    flag_cnt = 0;
    good_frame_cnt = 0;
    good_byte_cnt = 0;

    // Initialize error statistics
    crc_err_cnt = 0;
    abort_cnt = 0;
    seven_ones_cnt = 0;
    non_align_cnt = 0;
    giant_cnt = 0;
    runt_cnt = 0;
    

    // Loop through state machine, once per bit, 'till EOF
    while(get_bit(&next_bit) != EOF)
      {
        switch(state)
          {
            case HUNT:
              //fprintf(stderr, "State = HUNT\n");
              // Preload the first 7 bits to get things started
              byte = (byte >> 1) | (next_bit << 7);
              accumulated_bits++;
              if(accumulated_bits < 7)
                {
                  next_state = HUNT;
                }
              else
                {
                  next_state = IDLE;
                }
              break;

            case IDLE:
              //fprintf(stderr, "State = IDLE\n");
              byte = (byte >> 1) | (next_bit << 7);
              if(byte == FLAG)
                {
                  // Count it and keep hunting for more flags
                  flag_cnt++;
                  byte = 0x00;
                  accumulated_bits = 0;
                  next_state = HUNT;
                }
              else
                {
                  // A non-FLAG byte starts a frame
                  // Store the bits in the bit_buf, lsb first, and 
                  // change states.
                  for(i=0; i<8; i++)
                    {
                      bit_buf[i] = (byte >> i) & 0x01;
                    }
                  bit_buf_size = 8;
                  next_state = FRAMING;
                }
              break;

            case FRAMING:
              //fprintf(stderr, "State = FRAMING   bit_buf_size = %d\n", bit_buf_size);
              // Collect frame bits in bit_buf for later unstuffing
              if(bit_buf_size < BIT_BUF_MAX)
                {
                  bit_buf[bit_buf_size] = next_bit;
                  bit_buf_size++;
                }

              // Count consecutive 1 bits
              if(next_bit == 1)
                {
                  consecutive_one_bits++;
                }
              else
                {
                  consecutive_one_bits = 0;
                }

              // Test for Aborts and FLAGs
              if(consecutive_one_bits > 7)
                {
                  // Too many 1 bits in a row. Abort.
                  abort_cnt++;
                  seven_ones_cnt++;
                  byte = 0x00;
                  accumulated_bits = 0;
                  next_state = HUNT;
                }
              else
                {
                  // Pack bit into byte buffer and check for FLAG
                  byte = (byte >> 1) | (next_bit << 7);
                  if(byte != FLAG)
                    {
                      // Keep on collecting frame bits
                      next_state = FRAMING;
                    }
                  else 
                    {
                      // It's a FLAG. Frame is terminated.
                      flag_cnt++;
 
                      // Remove flag from bit_buf
                      bit_buf_size -= 8;

                      // Process bit_buf and
                      // see if we got a good frame.
                      status = unstuff(bit_buf_size, bit_buf, &frame_size, frame_buf);
                      //fprintf(stderr, "  Unstuffed Frame Size = %d\n", frame_size);
                      if(status == FAIL)
                        {
                          // Not an integer number of bytes.  Abort.
                          abort_cnt++;
                          non_align_cnt++;
                          //print_frame(frame_size, frame_buf);
                          //printf("    NON-ALIGNED FRAME\n\n");
                        }
                      else
                        {
                          // Check frame size
                          if(frame_size < 6)
                            {
                              // Too small
                              runt_cnt++;
                            }
                          else if(frame_size > FRAME_MAX)
                            {
                              // Too big
                              giant_cnt++;
                            }
                          else
                            {
                              // Size OK. Check crc
                              status = crc_valid(frame_size, frame_buf);
                              if(status == FAIL)
                                {
                                  // Log crc error
                                  crc_err_cnt++;
                                  //print_frame(frame_size, frame_buf);
                                  //printf("    BAD CRC\n\n");
                                }
                              else
                                {
                                  // Good frame! Log statistics
                                  good_frame_cnt++;
                                  good_byte_cnt += frame_size-2; // don't count CRC

                                  // Display decode
                                  printf("FRAME COUNT = %d\n", good_frame_cnt);
                                  decode_frame(frame_size, frame_buf);

                                  //route_packet(frame_size, frame_buf);
                                }
                            }
                        }
                      // Hunt for next flag or frame
                      byte = 0x00;
                      accumulated_bits = 0;
                      next_state = HUNT;
                    }
                 }     
              break;

          } // end switch

         state = next_state;
      }

    printf("Bitstream Statistics:\n");
    printf("    Flags    %6d\n", flag_cnt);
    printf("    Frames   %6d\n", good_frame_cnt);
    printf("    Bytes    %6d\n", good_byte_cnt);
    printf("Errors:\n");
    printf("    CRC Errs %6d\n", crc_err_cnt);
    printf("    Aborts   %6d\n", abort_cnt);
    printf("        7 Ones    %6d\n", seven_ones_cnt);
    printf("        Non Align %6d\n", non_align_cnt);
    printf("    Giants   %6d\n", giant_cnt);
    printf("    Runts    %6d\n", runt_cnt);


    return 0;
  }
