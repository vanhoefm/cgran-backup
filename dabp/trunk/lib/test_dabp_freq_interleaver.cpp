#include "dabp_freq_interleaver.h"
#include <iostream>
#include <iomanip>
using namespace std;

int main()
{
    dabp_freq_interleaver intlv(1,2000);
    int k;
    for (int n=0;n<1536;n++) {
        k=intlv.interleave(n);
        cout<<setw(5)<<(k>=1000?k-2000:k);
        if(n+1%8==0)
            cout<<endl;
    }
    return 0;
}

