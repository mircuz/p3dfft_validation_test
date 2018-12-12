/*
Validation test for a 2D FFT using p3dfft++.
The script is derived from the original 1D FFT example written in C.

This script does a 2D slab decomposition along y direction and perform an FFT on a
complex random dataset.

Setup and FFTs have timers to show how many time is spent to perform such actions.
Numbers of modes is selected during the declarations of the variables at top of
the program.

## How it works
Firstly the program generates a complex random dataset.
Once this is done the script does a 1D FFT along Z, followed by 1D FFT on X.
Everything is moved on an array called CONV where it is possible to do operations.
After those operations the script perform a 1D FFT along X and later along Z.

The results check is performed through pointers in the first and last array.

Since the code use MPI_Scatter to spread data among the processors, the number
of ny modes should be divisible by the number of processors.

Feel free to modify everything, and in case you find a better solution to spread
the data across processors ( without the actual limitation of ny%size != 0 ) please
contact me at: 	mirco.meazzo@mail.polimi.it



Author: Mirco Meazzo

*/

#include "p3dfft.h"
#include <math.h>
#include <stdio.h>

void print_res(double *A,int *mydims,int *gstart, int *mo, int N);
void normalize(double *,long int,double);
double check_res(double *A,double *B,int *mydims);

main(int argc,char **argv)
{
  int rank,size;
  int gdims[3];
  int pgrid1[3];
  int proc_order[3];
  int nx,ny,nz;
  nx = ny = nz = 256;
  int mem_order[]  = { 1, 2, 0};
  int mem_order2[] = { 0, 2, 1};
  int i,j,k,x,y,z;
  int *ldimsz,*ldimsx;
  long int sizez,sizex;
  double *INx, *INz, *CONV, *OUTz, *FINz, *OUTx;
  Grid *grid1x,*grid2x,*grid1z,*grid2z;
  int glob_start_z[3], glob_start_x[3];
  int type_ids1,type_ids2;
  double mydiff, diff = 0.0;
  int pdims[2], dim, dim2, mydims[3];
  Plan3D trans_1, trans_2, trans_3, trans_4;
  FILE *fp;


  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  MPI_Comm p3dfft_comm;
  MPI_Comm_dup( MPI_COMM_WORLD, &p3dfft_comm);


  //Header
   if(rank == 0) {
	   printf("\n\n[=========================================================================]\n"
			   "[==========================P3DFFT Validation test=========================]\n"
			   "[=========================================================================]\n\n");
	   printf("Running on %d cores\n",size);
	   printf("2D FFT in Double precision\n(%d %d %d) grid\n\n",nx,ny,nz);
   }

   if (rank == 0)
	   if ( ny % size != 0 || size > ny ) {
		   printf("!!! BE CAREFULL THE NUMBER OF ny SHOULD BE DIVISIBLE BY NUMBER OF TASKS !!!\n\n"
				   "Aborting...\n\n\n\n");
		   return 1;
	   }


   // Broadcast input parameters
   MPI_Bcast(&nx,1,MPI_INT,0,p3dfft_comm);
   MPI_Bcast(&ny,1,MPI_INT,0,p3dfft_comm);
   MPI_Bcast(&nz,1,MPI_INT,0,p3dfft_comm);
   MPI_Bcast(&dim,1,MPI_INT,0,p3dfft_comm);
   MPI_Bcast(&mem_order,3,MPI_INT,0,p3dfft_comm);
   MPI_Bcast(&mem_order2,3,MPI_INT,0,p3dfft_comm);

  // Establish 2D processor grid decomposition, either by readin from file 'dims' or by an MPI default
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

   if(rank == 0)
      printf("Using processor grid %d x %d\n\n",pdims[0],pdims[1]);

  //---------------------------------------------- P3DFFT Init -------------------------------------------

  double timer_setup = 0.0;
  timer_setup -= MPI_Wtime();

  p3dfft_setup();

  //Set up 2 transform types
  type_ids1 = P3DFFT_CFFT_BACKWARD_D;
  type_ids2 = P3DFFT_CFFT_FORWARD_D;

  //Set up global dimensions of the grid
  gdims[0] = nx;
  gdims[1] = ny;
  gdims[2] = nz;

  for(i=0; i < 3;i++) 
    proc_order[i] = i;

  // Define the processor grid
  dim = 2;										// Transform direction z
  pgrid1[0] = pdims[0];
  pgrid1[1] = pdims[1];
  pgrid1[2] = 1;

  if (rank == 0)
      printf("Task per direction (%d,%d,%d)\n\n", pgrid1[0],pgrid1[1],pgrid1[2]);

  //Initialize initial and final grids, based on the above information
  grid1z = p3dfft_init_grid(gdims,pgrid1,proc_order,mem_order,p3dfft_comm);
  grid2z = p3dfft_init_grid(gdims,pgrid1,proc_order,mem_order,p3dfft_comm);
  grid1x = p3dfft_init_grid(gdims,pgrid1,proc_order,mem_order2,p3dfft_comm);
  grid2x = p3dfft_init_grid(gdims,pgrid1,proc_order,mem_order2,p3dfft_comm);

  //Planning backward z-trasformation
  trans_1 = p3dfft_plan_1Dtrans(grid1z,grid1x,type_ids1,dim,0);
  //Planning forward z-trasformation
  trans_4 = p3dfft_plan_1Dtrans(grid2x,grid2z,type_ids2,dim,0);

  //Determine local z-grid dimensions, so the BEFORE & AFTER transform array dimensions.
  ldimsz = grid1z->ldims;
  sizez = ldimsz[0]*ldimsz[1]*ldimsz[2]*2;
  printf("Local dimensions on rank %d: %d,%d,%d\n", rank, ldimsz[0], ldimsz[1], ldimsz[2]);

  for(i=0;i<3;i++)
    mydims[mem_order[i]] = ldimsz[i];

  //Allocate input and output z-arrays
  INz=(double *) malloc(sizeof(double)*sizez);
  OUTz= (double *) malloc(sizeof(double) *sizez);
  FINz=(double *) malloc(sizeof(double) *sizez);

  dim2 = 0;										// Transform direction x
  //Planning backward x-trasformation
  trans_2 = p3dfft_plan_1Dtrans(grid1x,grid2x,type_ids1,dim2,0);
  //Planning forward x-trasformation
  trans_3 = p3dfft_plan_1Dtrans(grid2x,grid2x,type_ids2,dim2,0);

  //Determine local x-grid dimensions, so the INNER transformations arrays dimensions.
  ldimsx = grid1x->ldims;
  sizex = ldimsx[0]*ldimsx[1]*ldimsx[2]*2;

  for(i=0;i<3;i++)
	  mydims[mem_order2[i]] = ldimsx[i];

  //Allocate input and output x-arrays
  INx=(double *) malloc(sizeof(double)*sizex);
  CONV= (double *) malloc(sizeof(double) *sizex);
  OUTx =(double *) malloc(sizeof(double) *sizex);

  timer_setup += MPI_Wtime();


  //-------------------------------------------- Memory filling ----------------------------------------
  //Compute the starting indexes for each processor
  for(i=0;i<3;i++) {
	  glob_start_z[i] = grid1z->glob_start[i];
	  glob_start_x[i] = grid1x->glob_start[i];
  }

  printf("\n\t --------------------|Modes on processor: %d|--------------------  \n"
      	"\t|\tLocal array: \t (%d,%d,%d) -> (%d,%d,%d)\t\t|\n"
  		"\t ---------------------------------------------------------------  \n"
  		  , rank, glob_start_z[0], glob_start_z[1], glob_start_z[2],
  		  glob_start_z[0] + ldimsz[0], glob_start_z[1] + ldimsz[1], glob_start_z[2]+ ldimsz[2]);

  //Generate Input
  double *p_in;
  p_in = INz;
  j = 0;
  double *V;
  V = (double*) malloc( gdims[1]*gdims[2]*gdims[0]*2* sizeof(double));

  for ( i = 0; i < gdims[1]*gdims[2]*gdims[0]*2; i++ ){
	  //V[i] = j++;
	  V[i] = rand() % 10;
  }

  //Send chunks of array V to all processors
  MPI_Scatter( V, sizez, MPI_DOUBLE, INz, sizez,  MPI_DOUBLE, 0, p3dfft_comm);


  //Print Entries (Use it only in case of extreme needing!)
/*  if (rank == 0)														// This portion of code is useful to check whether
	  printf("DISTRIBUTED DATA: \n");									// or not the array filling works properly
  print_res(INz,mydims,glob_start_z,mem_order,mydims[2]);				// To access this portion remove the aborting procedure
																		// after the header.

  double *ptr, A, B;
  ptr = INz;
  for ( y = 0; y < ldimsz[1]; y++)
	  for ( x = 0; x < ldimsz[0]; x++)
		  for ( z = 0; z < ldimsz[2]; z++){
			  A = *ptr;
			  ptr++;
			  B = *ptr;
			  ptr++;
			  printf("R: %f \t I: %f \n", A, B);
		  }
*/

  //---------------------------------------------- FFT Execution -------------------------------------------
  MPI_Barrier(p3dfft_comm);
  double timer_fft = 0.0;
  timer_fft -= MPI_Wtime();

  //Backward z-transform (z->x grid)
  p3dfft_exec_1Dtrans_double(trans_1,INz,INx);
  normalize(INx,(long int) ldimsx[0]*ldimsx[1]*ldimsx[2],1.0/((double) mydims[mem_order2[2]]));
  MPI_Barrier(p3dfft_comm);

  //Backward x-transform (x->x grid)
  p3dfft_exec_1Dtrans_double(trans_2,INx,CONV);
  normalize(CONV,(long int) ldimsx[0]*ldimsx[1]*ldimsx[2],1.0/((double) mydims[mem_order[0]]));
  MPI_Barrier(p3dfft_comm);

  //Forward x-transform (x->x grid)
  p3dfft_exec_1Dtrans_double(trans_3,CONV,OUTx);
  MPI_Barrier(p3dfft_comm);

  //Forward z-transform (x->z grid)
  p3dfft_exec_1Dtrans_double(trans_4,OUTx,OUTz);
  MPI_Barrier(p3dfft_comm);

  timer_fft += MPI_Wtime();

  //Print Outputs (Use it only in case of extreme needing!)
/* if (rank == 0)
            printf("OUTPUT DATA: \n");
          print_res(OUTz,mydims,glob_start_z,mem_order,mydims[2]);
*/
  //---------------------------------------------- Check Results -------------------------------------------
  mydiff = check_res(INz,OUTz,mydims);

  diff = 0.;
  MPI_Reduce(&mydiff,&diff,1,MPI_DOUBLE,MPI_MAX,0,p3dfft_comm);
  if(rank == 0){
    printf("\nMax. diff. = %lg\n",diff);
    printf("Time spent to setup: %f s\nTime spent to do FFT: %f s\n\nTotal time: %f s\n\n\n", timer_setup, timer_fft, timer_fft+timer_setup);
  }

  //-------------------------- Release memory and clean up P3DFFT++ grids and data ------------------------
  free(INz); free(OUTz); free(INx); free(CONV); free(OUTx); free(V);
  p3dfft_free_grid(grid1x); p3dfft_free_grid(grid2x); p3dfft_free_grid(grid1z); p3dfft_free_grid(grid2z);
  p3dfft_cleanup();

  MPI_Finalize();
}


//=============================================== Functions ===============================================
void normalize(double *A,long int size,double f)
{
  long int i;
  
  for(i=0;i<size*2;i++)
    A[i] = A[i] * f;

}

void print_res(double *A,int *mydims,int *gstart, int *mo, int N)
{
  int x,y,z;
  double *p;
  int imo[3],i,j;
  
  for(i=0;i<3;i++)
    for(j=0;j<3;j++)
      if(mo[i] == j)
	imo[j] = i;

  p = A;
  for(z=0;z < mydims[2];z++)
    for(y=0;y < mydims[1];y++)
      for(x=0;x < mydims[0];x++) {
	if(fabs(*p) > N *1.25e-4 || fabs(*(p+1))  > N *1.25e-4) 
    printf("(%d %d %d) %lg %lg\n",x+gstart[imo[0]],y+gstart[imo[1]],z+gstart[imo[2]],*p,*(p+1));
	p+=2;
      }
}


double check_res(double *A,double *B,int *mydims)
{
  int x,y,z;
  double *p1,*p2,mydiff;
  int i,j;
  p1 = A;
  p2 = B;
  
  mydiff = 0.;
  for(z=0;z < mydims[2];z++)
    for(y=0;y < mydims[1];y++)
      for(x=0;x < mydims[0];x++) {
	if(fabs(*p1 - *p2) > mydiff)
	  mydiff = fabs(*p1-*p2);
	p1++;
	p2++;
      }
  return(mydiff);
}
