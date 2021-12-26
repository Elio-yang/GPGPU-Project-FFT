/*
MIT License

Copyright (c) 2021 Yang Yang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
/*
 * @author Elio Yang
 * @email  jluelioyang2001@gamil.com
 * @date 2021/12/22
 */
#include <iostream>
#include <vector>
#include <complex>
#include "tools.h"
#include <chrono>
using namespace std;


#define maxn 1000
#define pi acos(-1.0)

complex<double> A[maxn];
complex<double> B[maxn];
complex<double> C[maxn];
class Polynomial {
public:
        // the coeff from a0 to a_order
        // a0 + a1*x + a2*x^2 +...
        int order;
        int coeff[maxn];
        Polynomial(int n, vector<int> &coeffs) : order(n)
        {
                if (coeffs.size() != n + 1) {
                        cerr << "Not enough coeffs for this polyn" << endl;
                }
                int j=0;
                for (auto i: coeffs) {
                        coeff[j++]=i;
                }
#ifdef DEBUG
                cout << "Polyn [order " << n << "] created with [" << n + 1 << " coeffs]:";

                for (int i = 0; i < order + 1; i++) {
                        cout << coeff[i] << " ";
                }
                cout << endl;
#endif
        }
        void info()
        {
                cout << "================================================" << endl;
                cout << "Polyn [>>> order " << order << "]" << endl;
                cout << "Polyn [>>> coeffs " << order+1 << "] " << endl;
                for (int i = 0; i <= order; i++) {
                        cout << coeff[i] << " ";
                }
                cout << endl;
                cout << "f(x)=" << coeff[0] << "+";
                for (int i = 1; i <= order - 1; i++) {
                        cout << coeff[i] << "x^" << i << "+";
                }
                cout << coeff[order] << "x^" << order << endl;
                cout << "================================================" << endl;
        }

};

uint32_t next_power_two(uint32_t x){
        --x;
        x|=x>>1;
        x|=x>>2;
        x|=x>>4;
        x|=x>>8;
        x|=x>>16;
        return ++x;
}


#ifdef DEBUG
int cnt=0;
void printIndent(int n){
        for(int i=0;i<n;i++){
                printf("        ");
        }
}
#endif



// given a polyn coeff represent a=[a0,...,an-1] size=n
// return DFT of a y=[y0,...,yn-1] size = n
// size must be 2^n

// flg=-1 --> iFFT
void rFFT(complex<double> a[], unsigned N, int flg){
#ifdef DEBUG
        int thislevel=cnt;
        printIndent(thislevel);
        printf("[N is %d]\n",N);
        cnt++;
#endif
        if (N == 1){
#ifdef DEBUG
                printIndent(--cnt);
                printf("N is %d\n",N);
#endif

                return;
        }
        int half = N >> 1;
        complex<double> a0[half];
        complex<double> a1[half];
        for(int i=0; i < N; i+=2){
                a0[i/2]=a[i];
                a1[i/2]=a[i+1];
        }

#ifdef DEBUG
        printIndent(thislevel);
        printf("a :");
        for (int i = 0; i < N ; i++) {
                cout<<a[i]<<" ";
        }
        printf("\n");
        printIndent(thislevel);
        printf("a0 :");
        for (int i = 0; i < N / 2; i++) {
                cout<<a0[i]<<" ";
        }
        printf("\n");
        printIndent(thislevel);
        printf("a1 :");
        for (int i = 0; i < N / 2; i++) {
                cout<<a1[i]<<" ";
        }
        printf("\n");
#endif

        rFFT(a0,half,flg);
        rFFT(a1,half,flg);
        complex<double> wn = complex<double>(cos(2 * pi / N), flg*sin(2*pi/N));
        complex<double> w = complex<double>(1,0);
        for(int k=0;k<=half-1;k++){
                complex<double> u = w*a1[k];
                a[k]=a0[k]+u;
                a[k+half]=a0[k]-u;
                w *= wn;
        }
#ifdef DEBUG
        printIndent(thislevel);
        printf("half = %d ,>>>>>y :",half);
        for (int i = 0; i < N ; i++) {
                cout<<a[i]<<" ";
        }
         printf("\n");
#endif

#ifdef DEBUG
        printIndent(--cnt);
#endif
}

void FFT(complex<double>a[],unsigned N){
        rFFT(a,N,1);
}

void iFFT(complex<double>a[],unsigned N){
        rFFT(a,N,-1);
}



int main(int argc , char *argv[])
{

        int order_a = atoi(argv[1]);
        vector<int> poly_a_coeff = DataSet::genData(order_a + 1);
        int j=0;
        for(auto i:poly_a_coeff){
                A[j++].real(i);
        }
        j=0;

        int order_b = atoi(argv[2]);
        vector<int> poly_b_coeff = DataSet::genData(order_b + 1);
        for(auto i:poly_b_coeff){
                B[j++].real(i);

        }
        unsigned order_c = order_a + order_b;
        unsigned N = next_power_two(order_c);

        auto start = std::chrono::high_resolution_clock::now();
        FFT(A,N);
        FFT(B,N);

        for(int i=0;i<=N;i++){
                C[i] = A[i]*B[i];
        }
        iFFT(C,N);
        auto finish = std::chrono::high_resolution_clock::now();
        auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
        std::cout << microseconds.count() << std::endl;


        return 0;
}


