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
#include <random>
//#include <benchmark/benchmark.h>

#include "tools.h"

using namespace std;


#define DEBUG

#undef DEBUG





class Polynomial {
public:
        // the coeff from a0 to a_order
        // a0 + a1*x + a2*x^2 +...
        int order;
        vector<int> coeff;

        Polynomial(int n, vector<int> &coeffs) : order(n)
        {
                if (coeffs.size() != n + 1) {
                        cerr << "Not enough coeffs for this polyn" << endl;
                }
                for (auto i: coeffs) {
                        coeff.push_back(i);
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
                cout << "Polyn [>>> coeffs " << coeff.size() << "] " << endl;
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

        friend Polynomial &operator*(const Polynomial &a, const Polynomial &b)
        {
                int c_order = a.order + b.order;
                vector<int> c_coeff;
                c_coeff.reserve(c_order + 1);
                for (int j = 0; j <= c_order; j++) {
                        int cj = 0;
                        for (int k = 0; k <= j; k++) {
                                if (k >= a.coeff.size() || j - k >= b.coeff.size()) {
                                        cj += 0;
                                } else {
                                        cj += (a.coeff[k] * b.coeff[j - k]);
                                }
                        }
                        c_coeff.push_back(cj);
                }
                auto *res = new Polynomial(a.order + b.order, c_coeff);
                return *res;
        }

};

#ifdef cpuBench
// when do test eachtime just test one case
// otherwise there will be segmentation fault

vector<Polynomial> genPolynOrder10;
vector<Polynomial> genPolynOrder20;
vector<Polynomial> genPolynOrder40;
vector<Polynomial> genPolynOrder80;
vector<Polynomial> genPolynOrder100;

vector<vector<Polynomial>> dataPolyn={
        genPolynOrder10,
        genPolynOrder20,
        genPolynOrder40,
        genPolynOrder80,
        genPolynOrder100
};

int orders[]={10,20,40,80,100};

void preData(){
            for(int i=0;i<5;i++){
                    vector<Polynomial> &a = dataPolyn[i];
                    int order_a = orders[i];
                    // each with ten
                    for(int j=0;j<10;j++){
                            vector<int> coeff_j= DataSet::genData(order_a + 1);
                            Polynomial b(order_a, coeff_j);
                            a.push_back(b);
                    }
            }
}
static void bench_polyn_test_o10(benchmark::State & state){

        preData();

        for(auto _:state){
                // 10 polyns
                for(const auto& p :dataPolyn[0]){
                        Polynomial c= p*p;
                }
        }

}
BENCHMARK(bench_polyn_test_o10);


static void bench_polyn_test_o20(benchmark::State & state){
        preData();
        for(auto _:state){
                // 10 polyns
                for(const auto& p :dataPolyn[1]){
                        Polynomial c= p*p;
                }
        }
}
BENCHMARK(bench_polyn_test_o20);



static void bench_polyn_test_o40(benchmark::State & state){
        preData();

        for(auto _:state){
                // 10 polyns
                for(const auto& p :dataPolyn[2]){
                        Polynomial c= p*p;
                }
        }
}
BENCHMARK(bench_polyn_test_o40);



static void bench_polyn_test_o80(benchmark::State & state){
         preData();
        for(auto _:state){
                // 10 polyns
                for(const auto& p :dataPolyn[3]){
                        Polynomial c= p*p;
                }
        }
}
BENCHMARK(bench_polyn_test_o80);


static void bench_polyn_test_o100(benchmark::State & state){
        preData();
        for(auto _:state){
                // 10 polyns
                for(const auto& p :dataPolyn[4]){
                        Polynomial c= p*p;
                }
        }
}
BENCHMARK(bench_polyn_test_o100);

BENCHMARK_MAIN();
#endif

int main()
{
        int order_a = 8;
        vector<int> poly_a_coeff = DataSet::genData(order_a + 1);
        Polynomial a(order_a, poly_a_coeff);

        int order_b = 8;
        vector<int> poly_b_coeff = DataSet::genData(order_b + 1);
        Polynomial b(order_b, poly_b_coeff);

        Polynomial c = a * b;
        a.info();
        b.info();
        c.info();


        return 0;
}
