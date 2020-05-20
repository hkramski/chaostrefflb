#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

/**
 *  gcc -o mm -g -march=armv7 -mfpu=neon-vfpv4 matrix_matrix.c
 *  or cross compile
 *  arm-linux-gnueabihf-gcc -o mm -g -march=armv7 -mfpu=neon-vfpv4 matrix_matrix.c
 *
 * test with a = np.arange(4*n*m + 4m).reshape(4*n,4*m)
 * a.dot(a.T)
 * 
 * this file contains all the elements to show ARM7VL usage of 
 * SIMD architecture
 *
 */

void simple_transpose( float *, int, int, float *);
static inline void my_sgemm_4x4(float *, float *, float *,
				int, int, int );

/**
 * help routine (only for documentation demonstration
 * 
 * transpose some matrix matrix_a (size n_rows, n_cols) 
 * to matrix output
 *
 */

void simple_transpose( float *matrix_a, int n_rows_a, int n_cols_a,
		       float *output ) {
  for (int ra = 0; ra < n_rows_a; ra++)
    for (int ca = 0; ca < n_cols_a; ca++ )
      output[n_rows_a*ca+ra] = matrix_a[n_cols_a*ra+ca];
      //output[ra][ca] = matrix_a[ca][ra];
  return;
}


/**
 * Kernal function including the optimization and the 
 * 4 x 4 multiplication of 4 x4 fragments of large column based
 * matrixes matrix_a and matrix_b
 *
 * arguments:
 * - (float *) matrix_a: square matrix of size 4 x 4,
 * - (float *) matrix_b: square matrix of size 4 x 4,  
 * - (float *) output: 4 x 4 result (return value)
 * - (int) off_a,b,o: offset between to elements last element of one row
 *         and 1 element of next row matrix_a, matrix_b and output 
 * 
 * details documented here ![Using_SIMD](/home/eduard/work/wikiwhat/doc/Using_SIMD.md)
 */

static inline void my_sgemm_4x4(float *matrix_a, float *matrix_b,
				float *output,
				int off_a, int off_b, int off_o ) {
  /** \code */
  asm volatile (
    "# Start manual code \n\t"
    "# Matrix Multiplication \n\n\t"
    /* Maco section */
    ".macro  mul_col_f32 res_q, col0_d, col1_d\n\t"
    "vmla.f32 \\res_q, q8, \\col0_d[0]\n\t"
    "vmla.f32 \\res_q, q9, \\col0_d[1]\n\t"
    "vmla.f32 \\res_q, q10, \\col1_d[0]\n\t"
    "vmla.f32 \\res_q, q11, \\col1_d[1]\n\t"
    ".endm\n\n\t"
    /* end macro section */
    /* load current state of output -> q12 - q15 */
    "vld1.32 {q12}, [%6]!\n\t"
    "add %6, %6, %5\n\t"        /* add some offset until start of next row */
    "vld1.32 {q13}, [%6]!\n\t"
    "add %6, %6, %5\n\t"
    "vld1.32 {q14}, [%6]!\n\t"
    "add %6, %6, %5\n\t"
    "vld1.32 {q15}, [%6]!\n\t"
    /* load matrix_b (transposed!) -> q8 - q11 */
    "vld1.32 {q8}, [%2]!\n\t"
    "add %2, %2, %4\n\t"
    "vld1.32 {q9}, [%2]!\n\t"   
    "add %2, %2, %4\n\t"
    "vld1.32 {q10}, [%2]!\n\t"   
    "add %2, %2, %4\n\t"
    "vld1.32 {q11}, [%2]!\n\t"   
    /* load matrix_a -> q0 - q3 */
    "vld1.32 {q0}, [%1]!\n\t"   
    "add %1, %1, %3\n\t"
    "vld1.32 {q1}, [%1]!\n\t"   
    "add %1,%1, %3\n\t"
    "vld1.32 {q2}, [%1]!\n\t"   
    "add %1, %1, %3\n\t"
    "vld1.32 {q3}, [%1]!\n\t"
    /* end load registers
     * start doing the actual matrix multiplication as defined in macro */
    "mul_col_f32 q12, d0, d1\n\t"
    "mul_col_f32 q13, d2, d3\n\t"
    "mul_col_f32 q14, d4, d5\n\t"
    "mul_col_f32 q15, d6, d7\n\n\t"
    /* store the result [q12 - 115] into output */
    "vst1.32 {q12}, [%0]!\n\t"
    "add %0, %0, %5\n\t"
    "vst1.32 {q13}, [%0]!\n\t"
    "add %0, %0, %5\n\t"
    "vst1.32 {q14}, [%0]!\n\t"
    "add %0, %0, %5\n\t"
    "vst1.32 {q15}, [%0]!\n\t"
    /* start argument section of inline assembler */
    :"+r"((long) output)
    :"r"(&matrix_a[0]),"r"(&matrix_b[0]),"r"(off_a),"r"(off_b),
     "r"(off_o),"r"(&output[0]));
  /** \endcode */
  return;
}

/**
 * matrix matrix multiplication of some matrix_a and some matrix_b
 * (works only for size 4*n x 4*m)
 * the order is column based and output = a x b.transpose()
 *
 * the multiplication based on patch-wise standard multiplication algorithm
 * each patch of size 4 x 4
 */

void my_sgemm(float *matrix_a, int n_rows_a, int n_cols_a,
	      float *matrix_b, int n_rows_b, int n_cols_b,
	      float *output ) {
  int offset_a = 4*(n_cols_a-4);
  int offset_b = 4*(n_cols_b-4);
  for(int i=0;i<n_rows_a;i = i+4 ) {
    for(int j=0;j<n_cols_b;j = j+4 ) {    
      for(int k=0;k<n_cols_a;k = k+4 ) {    
	my_sgemm_4x4(&matrix_a[n_cols_a*i+k],
		     &matrix_b[n_cols_b*k+j], 
		     &output[n_cols_b*i+j],
		     offset_a, offset_b, offset_b);
      }    
    }
  }
  return;
}

/**
 * standard algorithm for matrix matrix multiplication
 * output = matrix_a.dot(matrix_bb.transpose())
 * - arguments:
 *  - (float *) a: column-based matrix size n_colums_a x n_rows_b
 *  - (int)     n_rows_a, n_cols_a: size of matrix a
 *  - (float *) b: column-based matrix size n_colums_a x n_rows_b
 *  - (int)     n_rows_b, n_cols_b: size of matrix b
 *  - (float *) output: column-based matrix = a.dot(b.T)
 * - return: void
 */

void simple_mm( float *a, int n_rows_a, int n_cols_a,
		float *b, int n_rows_b, int n_cols_b,
		float *output ) {
  for(int i=0;i<n_rows_a;i++)    
    for(int j=0;j<n_cols_b;j++) {    
      output[n_cols_b*i+j]=0;    
      for(int k=0;k<n_cols_a;k++) {    
	output[n_cols_b*i+j]+=a[n_cols_a*i+k]*b[n_cols_b*k+j];    
      }    
    }    
  return;
}

/**
 * int main():
 *     calls simple and optimized function and compare speed
 *     size defined by macros (N_ROWS_x, N_COLS_B: 4 * n, 4 * m)
 *     Matrix defined as:
 *     matrix_a = np.arange(N_ROWS_A*N_COLS_B).reshape(N_ROWS_A,N_COLS_A)
 *     matrix_b = a.T
 *     1. call optimized my_sgemm
 *     2. call simple_mm
 *     compare times, check identity
 *     print results
 * size of example matrix 
 */

#define N_COLS_A 256
#define N_ROWS_A 256
#define N_COLS_B N_ROWS_A
#define N_ROWS_B N_COLS_A

int main() {
  float matrix_a[N_ROWS_A][N_COLS_A];
  float matrix_b[N_ROWS_B][N_COLS_B];
  float matrix_aa[N_ROWS_A][N_COLS_A];
  float matrix_bb[N_ROWS_B][N_COLS_B];
  float buffer[N_ROWS_A][N_COLS_B];
  float reference[N_ROWS_A][N_COLS_B];
  struct timeval t1, t2, t3;
  long int durationf, durations;

  /**
   * matrix_a = np.arange(N_ROWS_A*N_COLS_B).reshape(N_ROWS_A,N_COLS_A)
   */

  for (int ra = 0; ra < N_ROWS_A; ra++) {
     for (int ca = 0; ca < N_COLS_A; ca++) {
       matrix_a[ra][ca] = N_COLS_A*ra+ca;
       matrix_aa[ra][ca] = N_COLS_A*ra+ca;
     }
  }
  
  /**
   * calculate matrix_b as matrix_a.T 
   *
   */
  
  simple_transpose(&matrix_a[0][0], N_ROWS_A, N_COLS_A,
  		   &matrix_b[0][0]);

  simple_transpose(&matrix_aa[0][0], N_ROWS_A, N_COLS_A,
  		   &matrix_bb[0][0]);

  /**
   * set outputs buffer (outpot of my_sgemm) and referece (simple_mm)
   * to zero
   */
  
  for (int ra = 0; ra < N_ROWS_A; ra++)
    for (int cb = 0; cb < N_COLS_B; cb++) {
      buffer[ra][cb] = 0.0;
      reference[ra][cb] = 0.0;
    }
  
  /**
   * 1. set timer to t1 (start of optimized algorithm)
   * 2. call optimized algorithm
   */
  
  gettimeofday(&t1, NULL);

  my_sgemm(&matrix_aa[0][0], N_ROWS_A, N_COLS_A,
  	   &matrix_bb[0][0], N_ROWS_B, N_COLS_B,
  	   &buffer[0][0]);
  
  /**
   * 3. set timer to t2 (end of optimized and start of simple algorithm)
   * 4. call optimized algorithm
   */
  
  gettimeofday(&t2, NULL);
  
  simple_mm(&matrix_a[0][0], N_ROWS_A, N_COLS_A,
	    &matrix_b[0][0], N_ROWS_B, N_COLS_B,
	    &reference[0][0]);
  
  /**
   * 3. set timer to t3 (end of simple algorithm)
   */
  
  gettimeofday(&t3, NULL);

  /**
   * calculate durations for optimized and simple algorithm
   */
  
  durationf = 1e6*(t2.tv_sec - t1.tv_sec)+(t2.tv_usec - t1.tv_usec);
  durations = 1e6*(t3.tv_sec - t2.tv_sec)+(t3.tv_usec - t2.tv_usec);

  /**
   * output (6 x 6 patch of result (both algorithm)
   */
  
  printf("my_sgemm\n");
  for (int ra=0; ra<6; ra++ ) {
    for (int cb=0; cb<6; cb++ ) printf("%.2e ", buffer[ra][cb]);
    printf("\n");
  }
  printf("reference\n");
  for (int ra=0; ra<6; ra++ ) {
    for (int cb=0; cb<6; cb++ ) printf("%.2e ", reference[ra][cb]);
    printf("\n");
  }

  /**
   * calculate mean sqare error 
   */
  
  float mse = 0.0F;
  for (int ra=0; ra<N_ROWS_A; ra++ ) {
    for (int cb=0; cb<N_COLS_B; cb++ ) {
      mse += (reference[ra][cb]-buffer[ra][cb]) *
	(reference[ra][cb]-buffer[ra][cb]);
    }
  }

  /**
   * print mse and ration of times optimized_time / simple_time
   */
  
  printf("MSE: %.5f [durationrate f/s %.5f]\n",mse,
	 (float)durationf/(float)durations);
  return 0;
}
