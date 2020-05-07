# ARMv7-l SIMD and using NEON

## Motivation for SIMD

<img src="pics/Adaline.jpg">

[(WikiCommons:Adaline)](https://commons.wikimedia.org/wiki/File:Adaline.jpg)


$\hat{y} = f(W\times \hat{x}+B)$

For a time series:

$X = \left[\hat{x_0},\hat{x_1},\dots,\hat{x_n}\right]$

We get:

$Y = f(W\times X+B)$

### That is what Thensorflow, numpy and lots of others are good about ...

## Matrix Mutlipication

Eor each element in the resulting matrix a scalar product of a specific column of the matrix W and a specifc row of matrix X is required. 

<img src="pics/mxm.png">

Now that takes a while:

Simple approach in C

~~~
   for (c = 0; c < m; c++) {
      for (d = 0; d < q; d++) {
        for (k = 0; k < p; k++) {
          sum = sum + first[c][k]*second[k][d];
        }
 
        multiply[c][d] = sum;
        sum = 0;
      }
    }
~~~

That requires for matrixs (100,100) x (100,1000) 100*100*1000 = 10 MFLOPS. What can be done to optimize the speed?

## Techniques to optimize the calculation

### Only splitting the Matrix

Two reasons:
1. optimize cache usage (**not today**)
2. using **SIMD power**

<img src="pics/matrix.png">


# SIMD (single instruction multiple data)

### Just a few words to Inlining Assembler in C (or C++)

Assembler [examples see](https://www.ibiblio.org/gferg/ldp/GCC-Inline-Assembly-HOWTO.html)

The most simple one, works on x86:

~~~
#include <stdio.h>

int main(void)
{
        int foo = 10, bar = 15;
        asm volatile ("addl  %%ebx,%%eax"
                      :"=a"(foo)
                      :"a"(foo), "b"(bar));
        printf("foo+bar=%d\n", foo);
        return 0;
}
~~~

From the gcc manual
~~~
asm asm-qualifiers ( AssemblerTemplate 
                 : OutputOperands 
                 [ : InputOperands
                 [ : Clobbers ] ])
~~~

## My sources

All points are from the link above The 
[NEON TM Version: 1.0 Programmer’s Guide](https://static.docs.arm.com/den0018/a/DEN0018A_neon_programmers_guide_en.pdf)

- general idea behind SIMD (not talking about MIMD)
- ARM NEON comparision with others (1.2 - pp 1-4)
- the instruction timing is not clear - depends as all calculations mainly on data fetching time.  
- Fundamentals of NEON technology (1.4 - pp 1-10)
  - 1.4.1 Registers q, d, s
  - 1.4.2 Datatypes




### What is it

With a single instruction a vector (or other structurs) can be calculated in parallel.

One assembler instruction multi/adds vectors of 4x4:

<img src="pics/matrix-simd.png">

Each of the 9 patches requires 4 x 4 = 16 SIMD instruction (compared to 4 x 4 x 4 = 64 ops ) fmla.f32. (multipy/Add)

## Remark about this document

This study is only for a better understanding of the SIMD instructions and SIMD performance of
the ARMV7-A core (actually this one is a CORTEX-A53, but the OS supports only the 32 bit
alternative.)

## Documents and Sources

[ARM ® and Thumb ® -2 Instruction Set](http://infocenter.arm.com/help/topic/com.arm.doc.qrc0001m/QRC0001_UAL.pdf)

[ARM Architecture Reference Manual ARMv7-A and ARMv7-R](https://static.docs.arm.com/ddi0406/c/DDI0406C_C_arm_architecture_reference_manual.pdf)

The 
[NEON TM Version: 1.0 Programmer’s Guide](https://static.docs.arm.com/den0018/a/DEN0018A_neon_programmers_guide_en.pdf) provides all the information required to realy do SIMD on ARMV7-A and R. 
The document explains the register structure of the single, double and 128 bit registers as well as the instructions. 

Besides other examples (Swapping color channel, FIR, 
cross product), 
there is also an example for matrix matrix multiplication. 

The example examined here is based on this document and the 4 x 4 matrix multiplication given (chapter 7.1, pp. 115.)

## About the example: my_sgemm

The matrix matrix multiplication calculates patches of 4 x 4 at one time the rest of the
calculation is straight forward.

~~~
for (i ...)
  for (j ...)
     for (k ...)
~~~
the inner loop calls the optimized 4 x 4 multiplication.


## Shape of the matrixes

All  matrixes in C are column-based. matrix_a is regular and matrix_b is transposed. (Therefore, all scalar products
of columns [B] with rows of [A] are column $\times$ column multipilications.)

The calculation is performing

$C = A \times B^\mathsf{T} + C$

Assuming the matrix A contains n rows and m columns, then
the element A[i,j] has in the c-array representing the matrix the index i * m + j.
If we want to extract a patch out of the matrix:
A[k:k+4,l:l+4], the for rows of the matrix could be calculated by,
- first row starts at k*m+l
- the next row starts with some offset o = m-4.
- same for the thrid and forth rows.

## The assembler SIMD part for the 4 x 4 multiplication

Purpose of the 4x4 matrix multiplication: It multiplies of a small 4 x 4 patch of some large 
colom-based matrixes, important to know: matrix_a is regular,
matrix_b is transposed.

~~~
static inline void my_sgemm_4x4(float *matrix_a, float *matrix_b,
                                float *output,
                                int off_a, int off_b, int off_o ) {
  /** \code */
  asm volatile (
    "# Start manual code \n\t"
    "# Matrix Multiplication \n\n\t"
~~~
Macro section
This macro performs the actual multiplication. It provides the output row for one column of matrix_a and the matrix_b (q8 - q11). The rows are stored in col0 and col1 (which corresponts to two 128 bit registers), the colums are stored in 
q8-q11. res_q gives the resulting output row.
~~~
    ".macro  mul_col_f32 res_q, col0_d, col1_d\n\t"
    "vmla.f32 \\res_q, q8, \\col0_d[0]\n\t"
    "vmla.f32 \\res_q, q9, \\col0_d[1]\n\t"
    "vmla.f32 \\res_q, q10, \\col1_d[0]\n\t"
    "vmla.f32 \\res_q, q11, \\col1_d[1]\n\t"
    ".endm\n\n\t"
~~~
End macro section

Start loading the 128 registers with 4 single floats. q12-q15 are first loaded 
with the current state of the output. 

After each register is loaded some
offset has to be added, since the next row starts with some offset. The same
mechanismus applies to all matrixes. 


load current state of output -> q12 - q15 */
~~~
    "vld1.32 {q12}, [%6]!\n\t"
    "add %6, %6, %5\n\t"        /* add some offset until start of next row */
    "vld1.32 {q13}, [%6]!\n\t"
    "add %6, %6, %5\n\t"
    "vld1.32 {q14}, [%6]!\n\t"
    "add %6, %6, %5\n\t"
    "vld1.32 {q15}, [%6]!\n\t"
~~~
load matrix_b (transposed!) -> q8 - q11 */
~~~
    "vld1.32 {q8}, [%2]!\n\t"
    "add %2, %2, %4\n\t"
    "vld1.32 {q9}, [%2]!\n\t"
    "add %2, %2, %4\n\t"
    "vld1.32 {q10}, [%2]!\n\t"
    "add %2, %2, %4\n\t"
    "vld1.32 {q11}, [%2]!\n\t"
~~~
load matrix_a -> q0 - q3
~~~
    "vld1.32 {q0}, [%1]!\n\t"
    "add %1, %1, %3\n\t"
    "vld1.32 {q1}, [%1]!\n\t"
    "add %1,%1, %3\n\t"
    "vld1.32 {q2}, [%1]!\n\t"
    "add %1, %1, %3\n\t"
    "vld1.32 {q3}, [%1]!\n\t"
~~~
End load registers

Start doing the actual matrix multiplication as defined in macro
~~~  
    "mul_col_f32 q12, d0, d1\n\t"
    "mul_col_f32 q13, d2, d3\n\t"
    "mul_col_f32 q14, d4, d5\n\t"
    "mul_col_f32 q15, d6, d7\n\n\t"
 ~~~
store the result [q12 - 115] into output
 ~~~
    "vst1.32 {q12}, [%0]!\n\t"
    "add %0, %0, %5\n\t"
    "vst1.32 {q13}, [%0]!\n\t"
    "add %0, %0, %5\n\t"
    "vst1.32 {q14}, [%0]!\n\t"
    "add %0, %0, %5\n\t"
    "vst1.32 {q15}, [%0]!\n\t"
~~~
start argument section of inline assembler
~~~
    :"+r"((long) output)
    :"r"(&matrix_a[0]),"r"(&matrix_b[0]),"r"(off_a),"r"(off_b),
     "r"(off_o),"r"(&output[0]));
  /** \endcode */
  return;
}
~~~

