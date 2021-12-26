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

#ifndef FFT_TOOLS_H
#define FFT_TOOLS_H
#include <random>
#include <vector>

using namespace std;


//[a,b]的随机整数，使用(rand() % (b-a+1))+ a;
#define random_low 10
#define random_hi  100
#define random_range (random_hi - random_low +1)
#define gen_a_random (rand()%(random_range)+ random_low)


static std::random_device rd;
static std::default_random_engine gen = std::default_random_engine(rd());
class DataSet {
public:
        // gen n random data
        static vector<int> genData(int n)
        {
                std::uniform_int_distribution<int> dis(random_low, random_hi);
                vector<int> gendata;
                gendata.reserve(n);
                for (int i = 0; i < n; i++) {
                        gendata.push_back(dis(gen));
                }
                return gendata;
        }

        static void showData(vector<int> &data)
        {
                cout << "Dataset with size " << data.size() << endl;
                for (auto i: data) {
                        cout << i << " ";
                }
                cout << endl;
        }
};
#endif //FFT_TOOLS_H
