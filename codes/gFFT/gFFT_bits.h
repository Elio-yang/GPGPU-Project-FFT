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

#pragma once
#include "cuda_related.h"
#include "gFFT_complex.h"

//////////////////////////////////////////////////////////////////////////////////
// Bits hack part
//////////////////////////////////////////////////////////////////////////////////
// to iterately do FFT
// the init. array must be reordered.
// 001-->100 like this 
// but considered the input index is a 32-b int
// shr is needed
/// @reoredered : the array after bits-ops
/// @device : the coeffs array 
/// @s      : log2(array-size) 
/// @threadN: threads for a block
/// restrict here for better parrallization 
__global__ void bits_rev(Complex* __restrict__ reordered, Complex* __restrict__ device, int s, size_t threadN) {
	int id = blockIdx.x *threadN + threadIdx.x;
	reordered [__brev(id) >> (32 - s)] = device [id];
}

uint32_t next_power_two(uint32_t x) {
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}
/////////////////////////////////////////////
// __brev test
/////////////////////////////////////////////
__global__  void inv(int* num) {
	int val = *num;
	val = __brev(val);
	*num = val;
}
void inv_test() {
	int * dnum;
	int * hnum = (int*)malloc(sizeof(int));
	*hnum = 3;
	cudaMalloc((void**)&dnum, sizeof(int));
	cudaMemcpy(dnum, hnum, sizeof(int), cudaMemcpyHostToDevice);
	inv << <1, 1 >> > (dnum);
	cudaMemcpy(hnum, dnum, sizeof(int), cudaMemcpyDeviceToHost);
	printf("%0x\n", *hnum);
}

