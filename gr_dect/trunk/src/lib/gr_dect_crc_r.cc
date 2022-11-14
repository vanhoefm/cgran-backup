/**
 * \file r-crc.c
 * Functions and types for CRC checks.
 *
 * Generated on Tue Jan  8 16:11:54 2008,
 * by pycrc v0.6.3, http://www.tty1.net/pycrc/
 * using the configuration:
 *    Width        = 16
 *    Poly         = 0x0589
 *    XorIn        = 0x0000
 *    ReflectIn    = False
 *    XorOut       = 0x0001
 *    ReflectOut   = False
 *    Algorithm    = table-driven
 *****************************************************************************/
#include "gr_dect_crc_r.h"
#include <stdint.h>
#include <stdlib.h>

/**
 * Static table used for the table_driven implementation.
 *****************************************************************************/
static const crc_t crc_table[16] = {
    0x0000, 0x0589, 0x0b12, 0x0e9b, 0x1624, 0x13ad, 0x1d36, 0x18bf,
    0x2c48, 0x29c1, 0x275a, 0x22d3, 0x3a6c, 0x3fe5, 0x317e, 0x34f7
};



/**
 * \brief          Update the crc value with new data.
 * \param crc      The current crc value.
 * \param data     Pointer to a buffer of \a data_len bytes.
 * \param data_len Number of bytes in the \a data buffer.
 * \return         The updated crc value.
 *****************************************************************************/
crc_t crc_update_tb(crc_t crc, const unsigned char *data, size_t data_len)
{
    unsigned int tbl_idx;

    while (data_len--) {
        tbl_idx = (crc >> 12) ^ (*data >> 4);
        crc = crc_table[tbl_idx & 0x0f] ^ (crc << 4);
        tbl_idx = (crc >> 12) ^ (*data >> 0);
        crc = crc_table[tbl_idx & 0x0f] ^ (crc << 4);

        data++;
    }
    return crc & 0xffff;
}


crc_t crc_update(crc_t crc, const unsigned char *data, size_t data_len)
{
    unsigned int i;
    bool bit;
    unsigned char c;

    while (data_len--) {
        c = *data++;
        for (i = 0x80; i > 0; i >>= 1) {
            bit = crc & 0x8000;
            if (c & i) {
                bit = !bit;
            }
            crc <<= 1;
            if (bit) {
                crc ^= 0x0589;
            }
        }
        crc &= 0xffff;
    }
    return crc & 0xffff;
}


unsigned int gr_crc_r(const unsigned char *buf)
{

    crc_t crc;
    crc = crc_init();
    crc = crc_update_tb(crc, buf, sizeof(buf) - 1);
    crc = crc_finalize(crc);
    return (unsigned int)crc;

}

unsigned int gr_crc_r(const std::string s)
{
    crc_t crc;
    crc = crc_init();
    crc = crc_update_tb(crc, (const unsigned char *) s.data(), s.size());
    //printf("ciriccio1 0x%lx\n", (long)crc);
    //crc = crc_update(crc, (const unsigned char *) s.data(), s.size());
    //printf("ciriccio2 0x%lx\n", (long)crc);
    crc = crc_finalize(crc);
    //printf("ciriccio3 0x%lx\n", (long)crc);
    return (unsigned int)crc;
 
}
