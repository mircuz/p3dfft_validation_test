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

   	printf("\n ----------------|Modes on processor: %d|----------------  \n", rank);
   	printf("|\tLocal array input: \t (%d,%d,%d) -> (%d,%d,%d)\t|\n",
   			istart[0], istart[1], istart[2], iend[0], iend[1], iend[2] );
   	printf("|\tLocal array output: \t (%d,%d,%d) -> (%d,%d,%d)\t|\n",
   			fstart[0], fstart[1], fstart[2], fend[0], fend[1], fend[2]);
   	printf(" -------------------------------------------------------  \n\n");

   MPI_Barrier( MPI_COMM_WORLD);


//------------------------------------- Allocazione della memoria LOCALE -----------------------------------

   double *A,*B, *C, *D, *p1, *p2, *p3, *p4;
   A = (double *) malloc(sizeof(double) * isize[0]*isize[1]*isize[2]);
   B = (double *) malloc(sizeof(double) * fsize[0]*fsize[1]*fsize[2]*2);
   C = (double *) malloc(sizeof(double) * isize[0]*isize[1]*isize[2]);
   D = (double *) malloc(sizeof(double) * fsize[0]*fsize[1]*fsize[2]*2);

   if (rank == 0) {
	   printf("Memory allocated\n");
   }


//-------------------------------------------- Riempimento memoria ----------------------------------------

   int x,y,z;
   double val_r, val_i;

   p1 = A;
   p2 = B;
   p3 = C;
   p4 = D;

  for(z=0;z < fsize[2];z++) {
        for(y=0;y < fsize[1];y++) {
          for(x=0;x < fsize[0];x++) {
       	  val_r = 1;
       	  val_i = x+y+z;
          *p2++ = val_r;
          *p2++ = val_i;
    //      *p4++ = 0.0;
    //      *p4++ = 0.0;

          }
        }
  }


//----------- Printf B------------------
  for (int i = 0; i < fsize[0]*fsize[1]*fsize[2]*2; i++){
	  if (i %2 == 0){
		  printf("R= %f \t ", B[i]);
	  }
	  else{
		  printf("I= %f \t rank = %d \n", B[i], rank);
	  }
  }


//=============================================== END SETUP ==============================================

//================================================== FFT =================================================
//---------------------------------------------- R2C #0 FFT ----------------------------------------------

   double f0time, f1time, normtime, b0time, factor;
   f0time = 0.0;
   unsigned char oper_1[]="fnf", oper_2[]="fnf";	// Design trasformations
   long int total_f_grid_size, total_i_grid_size, total_modes;
   total_f_grid_size = fsize[0] * fsize[1] * fsize[2]*2;
   total_i_grid_size = isize[0] * isize[1] * isize[2];
   total_modes = nx * ny * nz;


   if (rank == 0) {
      	printf("Operation on %dx%dx%d grid per %dx%dx%d arrays completed in %lgs"
      			", plus %lgs to normalize results\n",
				isize[0], isize[1], isize[2], nx/isize[0], ny/isize[1], nz/isize[2] ,f0time, normtime);
      	printf("\nStarting FFT (C2R)...\n");
      	printf("\nStarting FFT (R2C) ...\n"
      			"Transformation kind:\t z,y,x = %s \n", oper_2);
   }


//---------------------------------------------- C2R #0 FFT ----------------------------------------------

   MPI_Barrier(MPI_COMM_WORLD);
   b0time = b0time - MPI_Wtime();

   Cp3dfft_btran_c2r(B,C,oper_2);

   b0time = b0time + MPI_Wtime();

   MPI_Barrier(MPI_COMM_WORLD);

   // normalize
   normtime = 0.0;
   normtime = normtime - MPI_Wtime();

   factor = 1.0/total_modes;
   mult_array(C,total_i_grid_size, factor);

   normtime = normtime + MPI_Wtime();


/*/----------- Printf C------------------
   for (int i = 0; i < isize[0]*isize[1]*isize[2]; i++){
	   printf("R= %f \t rank = %d \n", C[i], rank);
   }
*/

   if (rank == 0) {
	   printf("Operation on %dx%dx%d grid per %dx%dx%d arrays completed in %lgs!\n",
			   fsize[0], fsize[1], fsize[2], nx/fsize[0], ny/fsize[1], nz/fsize[2]/2, b0time);
	   printf("!!Always check which mode exploit the symmetry!!\n");
	   printf("\nStarting FFT (R2C) ...\n"
	         	"Transformation kind:\t x,y,z = %s \n", oper_1);
   }


//---------------------------------------------- R2C #1 FFT ----------------------------------------------

   f1time = 0.0;
   f1time = f1time - MPI_Wtime();

   Cp3dfft_ftran_r2c(C,D,oper_1);

   f1time = f1time + MPI_Wtime();

     if (rank == 0) {
        	printf("Operation on %dx%dx%d grid per %dx%dx%d arrays completed in %lgs"
        			", plus %lgs to normalize results\n\n",
  				isize[0], isize[1], isize[2], nx/isize[0], ny/isize[1], nz/isize[2] ,f1time, normtime);
     }

  // Free work space
  Cp3dfft_clean();


//----------- Printf D------------------
    for (int i = 0; i < fsize[0]*fsize[1]*fsize[2]*2; i++){
  	  if (i %2 == 0){
  		  printf("R= %f \t ", D[i]);
  	  }
  	  else{
  		  printf("I= %f \t rank = %d \n", D[i], rank);
  	  }
    }


//=============================================== END FFT ===============================================

//============================================ CHECK RESULTS ============================================

  double cdiff,ccdiff, prec;

  cdiff = 0.0;

  for (int i = 0; i < fsize[0]*fsize[1]*fsize[2]*2; i++){
	 if(cdiff < fabs(D[i] - B[i])) {
         cdiff = fabs(D[i] - B[i]);
         printf("max difference until now is = %.20lf on rank = %d \n", cdiff, rank);
	 }
  }

   MPI_Reduce(&cdiff,&ccdiff,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

  if(rank == 0) {

    prec = 1.0e-14;

    if(ccdiff > prec * total_modes*0.25)
      printf("\nResults are incorrect\n");
    else
      printf("\nResults are correct\n");

    printf("max diff =%g\n",ccdiff);
  }

  free(A); free(B); free(C); free(D);


  MPI_Finalize();
  return 0;

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

