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
#include <thrust/complex.h>
//////////////////////////////////////////////////////////////////////////////////
// Complex part
//////////////////////////////////////////////////////////////////////////////////

#define PI acos(-1.0) 
typedef float2 Complex;
static __host__ __device__ inline Complex complexAdd(Complex a, Complex b) {
	Complex c;
	c.x = a.x + b.x;
	c.y = a.y + b.y;
	return c;
}
static __host__ __device__ inline Complex negativeComplex(Complex a) {
	Complex c;
	c.x = -a.x;
	c.y = -a.y;
	return c;
}
/*
(a+bi)(c+di)= ac+adi+bci-bd
*/
static __host__ __device__ inline Complex complexMult(Complex a, Complex b) {
	Complex c;
	c.x = a.x*b.x - a.y*b.y;
	c.y = a.x*b.y + a.y*b.x;
	return c;
}
