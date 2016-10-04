//  g++ -O3 -DNDEBUG -fPIC -shared testkernel.cpp  -o test.so

#include <iostream>

using namespace std;



extern "C"
{

    struct Field {
        double * data;
        long dim;
        long * strides;
        long * shape;
    };

    void kernel(Field * src)
    {
        cout << "Dimension " << src->dim << endl;
        for( int i=0; i<src->dim; ++i ) {
            cout << i << " Stride: " << src->strides[i] << "   Shape: " << src->shape[i] << endl;
        }

        for( long y = 0; y < src->shape[1]; ++y )
        {
            for( long x = 0; x < src->shape[0]; ++x )
            {
                src->data[ y*src->strides[1] + x*src->strides[0] ]  *= 2.0;
            }
        }

    }
}