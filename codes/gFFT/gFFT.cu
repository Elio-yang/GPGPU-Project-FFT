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


#include "cuda_related.h"
#include "tools.h"
using namespace std;


#define maxn 10000000
Complex A [10005];
Complex B [10005];
Complex C [maxn];


//////////////////////////////////////////////////////////////////////////////////
// Core FFT part
//////////////////////////////////////////////////////////////////////////////////
/// see CLRS ch.30
/*
cpu iteration:
	for(half =1;half<N;half<<1){
		w_m;
		for(j=0;j<N;j+=2*half){
			for(k=0;k<half;k++){
				t
				a[j+k]
				a[j+k+half]
			}
		}
	}
*/
/// see CLRS ch.30
/// you'll get everything you want about FFT'
/// the implementations here is well cooperated with CLRS
__device__ void FFT_core_loop(Complex* __restrict__ reordered, int j, int k, int m, int N, int flag) {
	if (j + k + m / 2 < N) {
		Complex t, u;
		// w_m = cos-isin
		t.x = __cosf((2.0*PI*k) / (1.0*m));
		t.y = -__sinf(flag*(2.0*PI*k) / (1.0*m));

		t = complexMult(t, reordered [k + j + m / 2]);
		u = reordered [k + j];
		reordered [k + j] = complexAdd(u, t);
		reordered [k + j + m / 2] = complexAdd(u, negativeComplex(t));
	}
}
__global__ void FFT_core_loop_paralel(Complex* __restrict__ reordered, int m, int N, int threadN, int flag) {
	int j = (blockIdx.x *threadN + threadIdx.x)*m;
	for (int k = 0; k < m / 2; k++) {
		FFT_core_loop(reordered, j, k, m, N, flag);
	}
}
void FFT(Complex *__restrict__ data, size_t N, size_t threads,int flag) {
	size_t dataSize = N * sizeof(Complex);
	Complex *reoredred, *device;
	// allocate memory for gpu device
	cudaMalloc((void**)&reoredred, dataSize);
	cudaMalloc((void**)&device, dataSize);
	cudaMemcpy(device, data, dataSize, cudaMemcpyHostToDevice);
	// reorder the data
	// N must be 2^k
	// this will be ensured by the bits operation to N
	int s = log2(N);
	bits_rev <<< ceil(N / threads), threads >>> (reoredred, device,s,threads);
	cudaDeviceSynchronize();
	for (int i = 1; i <= s; i++) {
		int m = 1 << i;
		FFT_core_loop_paralel <<< ceil((float)N / m / threads), threads >> > (reoredred, m, N, threads,flag);
	}
	cudaDeviceSynchronize();
	// the ans also in reoredred
	cudaMemcpy(data, reoredred, dataSize, cudaMemcpyDeviceToHost);
	cudaFree(reoredred);
	cudaFree(device);
}


int main(int argc ,char* argv[]) {
	// generate polyn A
	int order_a = atoi(argv[1]);
	vector<int> poly_a_coeff = DataSet::genData(order_a + 1);
	int j = 0;
	for (auto i : poly_a_coeff) {
		A [j++].x = i;;
	}
	j = 0;
	// generate polyn B
	int order_b = atoi(argv[2]);
	vector<int> poly_b_coeff = DataSet::genData(order_b + 1);
	for (auto i : poly_b_coeff) {
		B [j++].x = i;
	}

	unsigned order_c = order_a + order_b;
	unsigned N = next_power_two(order_c);
	int threads = 512;

	auto start = std::chrono::high_resolution_clock::now();
	// do FFT with flag 1
	FFT(&A [0], N, threads, 1);
	FFT(&B [0], N, threads, 1);
	// get values
	for (int i = 0; i <= N; i++) {
		C [i] = complexMult(A [i], B [i]);
	}
	// do inverse-FFT with flag -1
	FFT(&C [0], N, threads, -1);
	auto finish = std::chrono::high_resolution_clock::now();
	auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
	std::cout << microseconds.count() << std::endl;

	while (0);
}
