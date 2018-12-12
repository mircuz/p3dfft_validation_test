# p3dfft++_validation_test

Validation test for a 2D FFT using p3dfft++.
The script is derived from the original 1D FFT example written in C.

This script does a 2D slab decomposition along y direction and perform an FFT on a complex random dataset.

Setup and FFTs have timers to show how many time is spent to perform such actions.
Numbers of modes is selected during the declarations of the variables at top of the program.

## How it works 
Firstly the program generates a complex random dataset.
Once this is done the script does a 1D FFT along Z, followed by 1D FFT on X.
Everything is moved on an array called CONV where it is possible to do operations.
After those operations the script perform a 1D FFT along X and later along Z.

The results check is performed through pointers in the first and last array.

Since the code use MPI_Scatter to spread data among the processors, the number of ny modes should be divisible by the number of processors.


