#include "p3dfft.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc,char **argv)
{
   void print_all(double *,long int,int,long int),mult_array(double *,long int,double);

//================================================= SETUP ===================================================
//--------------------------------------------- Inizializzo MPI ---------------------------------------------

   int size, rank;
   MPI_Init(&argc,&argv);
   MPI_Comm_size(MPI_COMM_WORLD,&size);
   MPI_Comm_rank(MPI_COMM_WORLD,&rank);
   if (rank == 0) {
	   printf("\n\n[=======================================================================]\n"
			   "[========================P3DFFT Validation test=========================]\n"
			   "[=======================================================================]\n");
	   printf("\n%d processors available\n", size);
   }


//------------------------------------------ Lettura numero di modi -----------------------------------------

   int nx, ny, nz, ndim, dims[2];
   FILE *fp;

   if(rank == 0) {
	   ndim = 2;
      if((fp=fopen("stdin", "r"))==NULL){
         printf("Cannot open file.\n\n"
        		 "ERROR!!\n"
        		 "Setting to default nx=ny=nz=128, ndim=2\n");
         nx=ny=nz=128;
      } else {
         fscanf(fp,"%d %d %d %d\n",&nx,&ny,&nz,&ndim);
         fclose(fp);
      }
     printf("\n(%d x %d x %d) modes on %dD dimension\n", nx, ny, nz, ndim);
   }

   MPI_Bcast( &nx, 1, MPI_INT, 0, MPI_COMM_WORLD);				// Leggiamo il file solo da rank 0,
   MPI_Bcast( &ny, 1, MPI_INT, 0, MPI_COMM_WORLD);				// Broadcast x comunicare i dati al
   MPI_Bcast( &nz, 1, MPI_INT, 0, MPI_COMM_WORLD);				// resto della rete.
   MPI_Bcast( &ndim, 1, MPI_INT, 0, MPI_COMM_WORLD);


//---------------------------------------- Lettura griglia processori --------------------------------------

   if(ndim == 1) {
     dims[0] = 1; dims[1] = size;
   }
   else if(ndim == 2) {
     fp = fopen("dims","r");
     if(fp != NULL) {
       if(rank == 0)
         printf("\nReading proc. grid from file dims...\n");
       fscanf(fp,"%d %d\n",dims,dims+1);
       fclose(fp);
       if(dims[0]*dims[1] != size)
          dims[1] = size / dims[0];
     }
     else {
       if(rank == 0)
          printf("\nCreating proc. grid with mpi_dims_create\n");
       dims[0]=dims[1]=0;
       MPI_Dims_create(size,2,dims);
       if(dims[0] > dims[1]) {
          dims[0] = dims[1];
          dims[1] = size/dims[0];
       }
     }
   }

   if(rank == 0)
      printf("Using processor grid %d x %d\n\n ",dims[0],dims[1]);


//------------------------------------------ Inizializzazione P3DFFT ---------------------------------------

   if (rank == 0)
	   printf("Setting up P3DFFT...\n");

   int memsize[3];
   Cp3dfft_setup( dims, nx, ny, nz, MPI_Comm_c2f( MPI_COMM_WORLD ), nx, ny, nz, 0, memsize);

   // Get dimensions for input array - real numbers, X-pencil shape.
   // Note that we are following the Fortran ordering, i.e.
   // the dimension  with stride-1 is X. */
   int istart[3],isize[3],iend[3];
   Cp3dfft_get_dims( istart, iend, isize, 1);

   // Get dimensions for output array - complex numbers, Z-pencil shape.
   // Stride-1 dimension could be X or Z, depending on how the library
   // was compiled (stride1 option)
   int fstart[3],fsize[3],fend[3];
   Cp3dfft_get_dims( fstart, fend, fsize, 2);

   if( rank ==0 ) {
   	printf("\n ----------|Modes array per processor|----------  \n");
   	printf("|\tx_in= %d \t x_out= %d \t\t|\n", isize[0], fsize[0]);
   	printf("|\ty_in= %d \t y_out= %d \t\t|\n", isize[1], fsize[1]);
   	printf("|\tz_in= %d \t z_out= %d \t\t|\n", isize[2], fsize[2]);
   	printf(" -----------------------------------------------  \n\n");

   }


//----------------------------------------- Allocazione della memoria --------------------------------------

   double *A,*B, *C,*p1,*p2,*p;
   A = (double *) malloc(sizeof(double) * isize[0]*isize[1]*isize[2]);
   C = (double *) malloc(sizeof(double) * isize[0]*isize[1]*isize[2]);
   B = (double *) malloc(sizeof(double) * fsize[0]*fsize[1]*fsize[2]*2);

   if (rank == 0)
	   printf("Memory allocated\n");

//-------------------------------------------- Riempimento memoria ----------------------------------------

   int i,j,k,x,y,z,nu;
   double val;

   p1 = A;
   p2 = C;

   for(z=0;z < isize[2];z++)
     for(y=0;y < isize[1];y++)
       for(x=0;x < isize[0];x++) {
    	  val = sin(x)*cos(x)*z;
          *p1++ = val;
          *p2++ = 0.0;
       }

   long int total_f_grid_size, total_i_grid_size,total_modes;
   total_f_grid_size = fsize[0] * fsize[1] * fsize[2]*2;
   total_i_grid_size = isize[0] * isize[1] * isize[2];
   total_modes = nx * ny * nz;

   if (rank == 0) {
   	printf("Array ready\n");
   	printf("\nStarting FFT (R2C)...\n");
   }


//=============================================== END SETUP ==============================================

//================================================== FFT =================================================
//---------------------------------------------- Forward FFT ---------------------------------------------

   double ftime, normtime, btime, factor;
   unsigned char oper_1[]="fft", oper_2[]="tff";	// Design trasformations
   ftime = 0.0;

   MPI_Barrier(MPI_COMM_WORLD);
   ftime = ftime - MPI_Wtime();

   Cp3dfft_ftran_r2c(A,B,oper_1);

   ftime = ftime + MPI_Wtime();

/*     if(rank == 0)
        printf("Result of forward transform\n");

     print_all(B,total_f_grid_size,rank,total_modes);
*/

// normalize
   normtime = 0.0;
   normtime = normtime - MPI_Wtime();

   factor = 1.0/total_modes;
   mult_array(B,total_f_grid_size,factor);

   normtime = normtime + MPI_Wtime();


   if (rank == 0) {
      	printf("Operation on %dx%dx%d grid per %dx%dx%d arrays completed in %lgs"
      			", plus %lgs to normalize results\n",
				isize[0], isize[1], isize[2], nx/isize[0], ny/isize[1], nz/isize[2] ,ftime, normtime);
      	printf("\nStarting FFT (C2R)...\n");
   }


   MPI_Barrier(MPI_COMM_WORLD);
   btime = btime - MPI_Wtime();

   Cp3dfft_btran_c2r(B,C,oper_2);

   btime = btime + MPI_Wtime();

   if (rank == 0) {
	   printf("Operation on %dx%dx%d grid per %dx%dx%d arrays completed in %lgs!\n",
			   fsize[0], fsize[1], fsize[2], nx/fsize[0], ny/fsize[1], nz/fsize[2]/2, btime);
	   printf("!!Always check which mode exploit the symmetry!!\n\n");
   }

  // Free work space
  Cp3dfft_clean();


//=============================================== END FFT ===============================================

//============================================ CHECK RESULTS ============================================

  double cdiff,ccdiff, prec;

  cdiff = 0.0; p2 = C;p1 = A;
  for(z=0;z < isize[2];z++)
    for(y=0;y < isize[1];y++)
       for(x=0;x < isize[0];x++) {
	 if(cdiff < fabs(*p2 - *p1)) {
           cdiff = fabs(*p2 - *p1);
	   /* 	   printf("x,y,z=%d %d %d,cdiff,p1,p2=%f %f %f\n",x,y,z,cdiff,*p1,*p2); */
	 }
          p1++;
          p2++;
        }

   MPI_Reduce(&cdiff,&ccdiff,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

  if(rank == 0) {

    prec = 1.0e-14;

    if(ccdiff > prec * total_modes*0.25)
      printf("Results are incorrect\n");
    else
      printf("Results are correct\n");

    printf("max diff =%g\n",ccdiff);
  }



  MPI_Finalize();

}


void mult_array(double *A,long int nar,double f)

{
  long int i;

  for(i=0;i < nar;i++)
    A[i] *= f;
}



void print_all(double *A,long int nar,int proc_id,long int Nglob)

{
  int x,y,z,conf,Fstart[3],Fsize[3],Fend[3];
  long int i;

  conf = 2;
  Cp3dfft_get_dims(Fstart,Fend,Fsize,conf);
  /*
  Fsize[0] *= 2;
  Fstart[0] = (Fstart[0]-1)*2;
  */
  for(i=0;i < nar;i+=2)
    if(fabs(A[i]) + fabs(A[i+1]) > Nglob *1e-2) {
      z = i/(2*Fsize[0]*Fsize[1]);
      y = i/(2*Fsize[0]) - z*Fsize[1];
      x = i/2-z*Fsize[0]*Fsize[1] - y*Fsize[0];

      printf("(%d,%d,%d) %.16lg %.16lg\n",x+Fstart[0],y+Fstart[1],z+Fstart[2],A[i],A[i+1]);

    }
}
