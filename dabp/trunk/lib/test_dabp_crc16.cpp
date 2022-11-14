#include "dabp_crc16.h"
#include <iostream>
#include <cstdlib>
using std::cout;
using std::endl;

int main()
{
    const int data_size=1104;
    dabp_crc16 crc;
    unsigned char x[data_size+2];
    int i;
    for(i=0;i<data_size;i++)
        x[i]=rand()&0xff;
    
    crc.generate(x,data_size);
    //x[0]=~x[0];
    cout<<crc.check(x,data_size)<<endl;
}

