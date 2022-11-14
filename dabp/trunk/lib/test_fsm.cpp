#include <fsm.h>
#include <iostream>

void show(const std::vector<int> & x);

int main()
{
    const int g[]={0133,0171,0145,0133};
    //std::vector<int> gv(g,g+4);
    fsm f(1,4,std::vector<int>(g,g+4));
    //show(gv);
    show(f.NS());
    return 0;
}

void show(const std::vector<int> & x)
{
    std::vector<int>::const_iterator i;
    for(i=x.begin();i!=x.end();i++)
        std::cout<<(*i)<<" ";
    std::cout<<std::endl;
}

