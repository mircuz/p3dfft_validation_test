/* P3DFFT++ test,
 * designed to check whether P3DFFT++ is better than an hand-made mpi-ed FFT or not.
 *
 * Author: Mirco Meazzo */

#include "p3dfft.h"
#include <math.h>
#include <stdio.h>

void init_wave(double *,int[3],int *,int[3]);
void print_res(double *,int *,int *,int *);
void normalize(double *,long int,int *);
double check_res(double*,double *,int *);
void write_buf(double *buf,char *label,int sz[3],int mo[3], int taskid);

//============================================== START SCRIPT ================================================
//============================================================================================================


main(int argc,char **argv)
{
  int i,j,k,x,y,z;
  FILE *fp;


//================================================= SETUP ===================================================
//------------------------------------------------ MPI Init -------------------------------------------------

  MPI_Init(&argc,&argv);
  int rank,size;
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);


//---------------------------------------------- Reading modes ---------------------------------------------

  int nx,ny,nz,ndim;

   if(rank == 0) {
	   printf("\n\n[=======================================================================]\n"
	   			   "[========================P3DFFT Validation test=========================]\n"
	   			   "[=======================================================================]\n\n");
	   	   printf("\n%d processors available\n", size);
     if((fp=fopen("stdin", "r"))==NULL){
        printf("Cannot open file. Setting to default nx=ny=nz=128, ndim=2, n=1.\n");
        nx=ny=nz=128; ndim=2;
     } else {
        fscanf(fp,"%d %d %d %d\n",&nx,&ny,&nz,&ndim);
        fclose(fp);
     }
     printf("\nWorking on (%dx%dx%d) modes\n", nx, ny, nz);
   }
   MPI_Bcast(&nx,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&ny,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nz,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&ndim,1,MPI_INT,0,MPI_COMM_WORLD);


//---------------------------------------- Read processors grid --------------------------------------

   int pdims[2];

   if(ndim == 1) {
     pdims[0] = 1; pdims[1] = size;
     if(rank == 0)
       printf("Using one-dimensional decomposition\n\n");
   }
   else if(ndim == 2) {
     if(rank == 0)
       printf("Using two-dimensional decomposition\n\n");
     fp = fopen("dims","r");
     if(fp != NULL) {
       if(rank == 0)
         printf("Reading proc. grid from file dims\n");
       fscanf(fp,"%d %d\n",pdims,pdims+1);
       fclose(fp);
       if(pdims[0]*pdims[1] != size)
          pdims[1] = size / pdims[0];
     }
     else {
       if(rank == 0)
          printf("Creating proc. grid with mpi_dims_create\n");
       pdims[0]=pdims[1]=0;
       MPI_Dims_create(size,2,pdims);
       if(pdims[0] > pdims[1]) {
          pdims[0] = pdims[1];
          pdims[1] = size/pdims[0];
       }
     }
   }

   if(rank == 0)
      printf("Using processor grid %d x %d\n",pdims[0],pdims[1]);


//---------------------------------------------- P3DFFT Init -------------------------------------------

   if (rank == 0)
   	   printf("\n\nSetting up P3DFFT++...\n");

  p3dfft_setup();


  //Set up transform types for 3D transforms

  int type_ids1[3];
  int type_ids2[3];

  type_ids1[0] = P3DFFT_CFFT_BACKWARD_D;
  type_ids1[1] = P3DFFT_CFFT_BACKWARD_D;
  type_ids1[2] = P3DFFT_CFFT_BACKWARD_D;

  type_ids2[0] = P3DFFT_CFFT_FORWARD_D;
  type_ids2[1] = P3DFFT_CFFT_FORWARD_D;
  type_ids2[2] = P3DFFT_CFFT_FORWARD_D;


  //Initialize 3D transforms

  Type3D type_1,type_2;

  type_1 = p3dfft_init_3Dtype(type_ids1);
  type_2 = p3dfft_init_3Dtype(type_ids2);

  if (rank == 0) {
	  printf("\n\t • Trasformations initializated\n");
  }


  //Set up GLOBAL dimensions of the INPUT & OUTPUT grids

  int gdims1[3], proc_order[3], mem_order[3];
  int p1,p2;

  gdims1[0] = nx;
  gdims1[1] = ny;
  gdims1[2] = nz;
  for(i=0; i < 3;i++) {
    proc_order[i] = mem_order[i] = i; // The simplest case of sequential ordering
  }

  if (rank == 0) {
  	  printf("\t • Grid global dimensions defined\n");
    }

  p1 = pdims[0];
  p2 = pdims[1];


  /* Define the initial processor grid. In this case, it's a 2D pencil,
  with 1st dimension local and the 2nd and 3rd split by iproc and jproc tasks respectively */

  int pgrid1[3];

  pgrid1[0] = 1;
  pgrid1[1] = p1;
  pgrid1[2] = p2;


  //Define the final processor grid

  int pgrid2[3];

  pgrid2[0] = 1;
  pgrid2[1] = p2;
  pgrid2[2] = p1;

  if (rank == 0) {
  	  printf("\t • Stride defined\n");
    }


  /* Set up the final global grid dimensions (these will be different from the original
   *  dimensions in one dimension since we are doing real-to-complex transform) */

  int gdims2[3];
  for(i=0; i < 3;i++) 
    gdims2[i] = gdims1[i];

  int mem_order2[] = {2,1,0};

  //Initialize initial and final grids, based on the above information

  Grid *grid1,*grid2;

  grid1 = p3dfft_init_grid(gdims1,pgrid1,proc_order,mem_order,MPI_COMM_WORLD);
  grid2 = p3dfft_init_grid(gdims2,pgrid2,proc_order,mem_order2,MPI_COMM_WORLD);

  if (rank == 0) {
    	  printf("\t • Grid initialized\n");
      }


  //Set up the forward transform, based on the predefined 3D transform type and grid1 and grid2.
  //This is the planning stage, needed once as initialization.

  Plan3D trans_1;
  trans_1 = p3dfft_plan_3Dtrans(grid1,grid2,type_1,0);

  if (rank == 0) {
    	  printf("\t • Handle trasformation 1 generated\n");
      }


  //Now set up the backward transform

  Plan3D trans_2;
  trans_2 = p3dfft_plan_3Dtrans(grid2,grid1,type_2,0);

  if (rank == 0) {
      	  printf("\t • Handle trasformation 2 generated\n");
        }


  /*Save the LOCAL array dimensions.
  Note: dimensions and global starts given by grid object are in physical coordinates,
  which need to be translated into storage coordinates: */

  long int size1;
  int ldims[3], glob_start[3];		/* Local dimensions vector of the first transformation and
  	  	  	  	  	  	  	  	  	  relative global starting coordinates vector of the local subgrid */
  for(i=0;i<3;i++) {
    glob_start[mem_order[i]] = grid1->glob_start[i];
    ldims[mem_order[i]] = grid1->ldims[i];
  }

  size1 = ldims[0]*ldims[1]*ldims[2];


  //Determine local array dimensions and allocate fourier space, complex-valued out array

  long int size2;
  int ldims2[3], glob_start2[3];

  for(i=0;i<3;i++) {
	  glob_start2[mem_order2[i]] = grid2->glob_start[i];
      ldims2[mem_order2[i]] = grid2->ldims[i];
  }

  size2 = ldims2[0]*ldims2[1]*ldims2[2];


  printf("\n\t --------------------|Modes on processor: %d|--------------------  \n"
    	"\t|\tGlobal array input: \t (%d,%d,%d) -> (%d,%d,%d)\t|\n"
		"\t|\tGlobal array output: \t (%d,%d,%d) -> (%d,%d,%d)\t|\n"
    	"\t|\tArray size: \t\t (%d,%d,%d) -> (%d,%d,%d)\t|\n"
		"\t ---------------------------------------------------------------  \n"
				, rank, glob_start[0], glob_start[1], glob_start[2],
				glob_start[0] + ldims[0], glob_start[1] + ldims[1], glob_start[2]+ ldims[2],
				glob_start2[0], glob_start2[1], glob_start2[2],
				glob_start2[0] + ldims2[0], glob_start2[1] + ldims2[1], glob_start2[2]+ ldims2[2],
				ldims[0], ldims[1], ldims[2], ldims2[0], ldims2[1], ldims2[2]);



   if (rank == size -1) {
	   printf("\n...P3DFFT++ setup completed!\n\n");
   }


//----------------------------------------- Memory allocation --------------------------------------
/* 1D Array pointers used during the trasformation MUST BE LOCAL portion of the global 3D array.
 * The input & output array's pointer containing the 3D grid stored contiguously in memory,
 *  based on the local grid dimensions and storage order of grid1 and grid2.
 */

  double *IN, *OUT, *FIN;

  IN =(double *) malloc(sizeof(double) *size1*2);
  FIN=(double *) malloc(sizeof(double) *size1*2);
  OUT=(double *) malloc(sizeof(double) *size2*2);

  if (rank == 0)
  	   printf("Memory allocated\n");


//-------------------------------------------- Memory assignment ----------------------------------------
//Initialize the IN array with a sine wave in 3D, based on the starting positions of my local grid within the global grid

  init_wave(IN,gdims1,ldims,glob_start);


//=============================================== END SETUP ==============================================

//================================================== FFT =================================================

  if (rank == 0) {
        	printf("Array ready\n");
        	printf("\nStarting Backward FFT ...\n");
        }

  int total_modes = gdims1[0]*gdims1[1]*gdims1[2];
  double timer = 0.0;


  // Transformation 1

  timer -= MPI_Wtime();
  p3dfft_exec_3Dtrans_double(trans_1,IN,OUT,0);
  timer += MPI_Wtime();

  MPI_Barrier( MPI_COMM_WORLD );

  if (rank == 0) {
  	  printf("Operation on %dx%dx%d grid per %dx%dx%d arrays completed in %lgs\n",
  			  ldims[0], ldims[1], ldims[2], nx/ldims[0], ny/ldims[1], nz/ldims[2] , timer);
  	  printf("\nStarting Forward FFT ...\n");
  }

/*
  if(rank == 0)
	  printf("\nResults of forward transform: \n");
  print_res(OUT,gdims1,ldims2,glob_start2);
*/
  double normtime = 0.0;
  normtime -= MPI_Wtime();
  normalize(OUT,size2,gdims1);
  normtime += MPI_Wtime();


  // Transformation 2

  timer -= MPI_Wtime();
  p3dfft_exec_3Dtrans_double(trans_2,OUT,FIN,0); // Backward (inverse) complex-to-real 3D FFT
  timer += MPI_Wtime();

  MPI_Barrier( MPI_COMM_WORLD );

  if (rank == 0) {
      	  printf("Operation on %dx%dx%d grid per %dx%dx%d arrays completed in %lgs, plus %lgs to normalize\n\n",
      			  ldims2[0], ldims2[1], ldims2[2], nx/ldims2[0], ny/ldims2[1], nz/ldims2[2] , timer, normtime);
      }


//=============================================== END FFT ===============================================

//============================================ CHECK RESULTS ============================================

  double mydiff;
  double diff = 0.0;

  mydiff = check_res(IN,FIN,ldims);
  diff = 0.;
  MPI_Reduce(&mydiff,&diff,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  if(rank == 0) {
    if(diff > 1.0e-14 * total_modes *0.25)
      printf("Results are incorrect\n");
    else
      printf("Results are correct\n");
    printf("Max. diff. =%lg\n",diff);
  }

  free(IN); free(OUT); free(FIN);

  // Clean up grid structures

  p3dfft_free_grid(grid1);
  p3dfft_free_grid(grid2);

  // Clean up all P3DFFT++ data

  p3dfft_cleanup();

  MPI_Finalize();
}


//=============================================== END SCRIPT ===============================================
//==========================================================================================================


void normalize(double *A,long int size,int *gdims1)
{
  long int i;
  double f = 1.0/(((double) gdims1[0])*((double) gdims1[1])*((double) gdims1[2]));
  
  for(i=0;i<size*2;i++)
    A[i] = A[i] * f;

}

void init_wave(double *IN,int *gdims1,int *ldims,int *gstart)
{
  double *sinx,*siny,*sinz,sinyz,*p;
  int x,y,z;
  double twopi = atan(1.0)*8.0;

  sinx = (double *) malloc(sizeof(double)*gdims1[0]);
  siny = (double *) malloc(sizeof(double)*gdims1[1]);
  sinz = (double *) malloc(sizeof(double)*gdims1[2]);

   for(z=0;z < ldims[2];z++)
     sinz[z] = sin((z+gstart[2])*twopi/gdims1[2]);
   for(y=0;y < ldims[1];y++)
     siny[y] = sin((y+gstart[1])*twopi/gdims1[1]);
   for(x=0;x < ldims[0];x++)
     sinx[x] = sin((x+gstart[0])*twopi/gdims1[0]);

   p = IN;
   for(z=0;z < ldims[2];z++)
     for(y=0;y < ldims[1];y++) {
       sinyz = siny[y]*sinz[z];
       for(x=0;x < ldims[0];x++) {
          *p++ = sinx[x]*sinyz;
          *p++ = 0.0;
       }
     }

   free(sinx); free(siny); free(sinz);
}

void print_res(double *A,int *gdims1,int *ldims,int *gstart)
{
  int x,y,z;
  double *p;
  double total_modes;
  int imo[3],i,j;
  
  total_modes = gdims1[0]*gdims1[1];
  total_modes *= gdims1[2];
  p = A;
  for(z=0;z < ldims[2];z++)
    for(y=0;y < ldims[1];y++)
      for(x=0;x < ldims[0];x++) {
	if(fabs(*p) > total_modes *1.25e-4 || fabs(*(p+1))  > total_modes *1.25e-4)
    printf("(%d %d %d) %lg %lg\n",x+gstart[0],y+gstart[1],z+gstart[2],*p,*(p+1));
	p+=2;
      }
}

double check_res(double *A,double *B,int *ldims)
{
  int x,y,z;
  double *p1,*p2,d,mydiff;
  int imo[3],i,j;
  p1 = A;
  p2 = B;
  

  mydiff = 0.;
  for(z=0;z < ldims[2];z++)
    for(y=0;y < ldims[1];y++)
      for(x=0;x < ldims[0];x++) {
	d = (*p1-*p2)*(*p1-*p2);
	p1++; p2++;
	d += (*p1-*p2)*(*p1-*p2);
	if(d > mydiff)
	  mydiff = d;
	p1++;
	p2++;
      }
  return(sqrt(mydiff));
}

