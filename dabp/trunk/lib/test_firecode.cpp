#include "dabp_firecode.h"
#include <iostream>
#include <cstdlib>
using std::cout;
using std::endl;

int main()
{
    /*unsigned char c[]={127,254};
    unsigned short s;
    s=(c[0]<<8)|c[1];
    cout<<s<<endl;
    */
    dabp_firecode fc;
    unsigned char x[]={1, 157, 112, 29, 83, 189, 128, 182, 236, 107, 199};
    int i;
    x[5]=22;
    cout<<"check:"<<endl;
    cout<<fc.check(x)<<endl;
    cout<<"encode:"<<endl;
    cout<<fc.encode(x)<<endl;
}

