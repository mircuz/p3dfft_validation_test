/*
This program exemplifies the use of 1D transforms in P3DFFT++. 1D transforms are performed on 3D arrays, in the dimension specified as an argument. This could be an isolated 1D transform or a stage in a multidimensional transform. This function can do local transposition, i.e. arbitrary input and output memory ordering. However it does not do an inter-processor transpose (see test_transMPI for that). 
!
! This program initializes a 3D array with a 3D sine wave, then
! performs forward real-to-complex transform, backward comples-to-real 
! transform, and checks that
! the results are correct, namely the same as in the start except
! for a normalization factor. It can be used both as a correctness
! test and for timing the library functions.
!
! The program expects 'stdin' file in the working directory, with
! a single line of numbers : Nx,Ny,Nz,dim,Nrep,MOIN(1)-(3),MOOUT(1)-(3). 
! Here Nx,Ny,Nz are 3D grid dimensions, dim is the dimension of 1D transform 
! (valid values are 0 through 2, and the logical dimension si specified, i.e. actual storage dimension may be different as specified by MOIN mapping), Nrep is the number of repititions. 
! MOIN are 3 values for the memory order of the input grid, valid values of each is 0 - 2, not repeating. Similarly, MOOUT is the memory order of the output grid. 
! Optionally a file named 'dims' can also be provided to guide in the choice
! of processor geometry in case of 2D decomposition. It should contain
! two numbers in a line, with their product equal to the total number
! of tasks. Otherwise processor grid geometry is chosen automatically.
! For better performance, experiment with this setting, varying
! iproc and jproc. In many cases, minimizing iproc gives best results.
! Setting it to 1 corresponds to one-dimensional decomposition.
!
! If you have questions please contact Dmitry Pekurovsky, dmitry@sdsc.edu
*/

#include "p3dfft.h"
#include <math.h>
#include <stdio.h>

void init_wave(double *,int[3],int *,int[3]);
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
  nx = ny = nz = 3;
  int mem_order[]  = { 1, 2, 0};
  int mem_order2[] = { 0, 2, 1};
  int i,j,k,x,y,z;
  int *ldimsz,*ldimsx;
  long int sizez,sizex;
  double *INx, *INz, *CONV, *OUTz, *FINz, *OUTx;
  Grid *grid1x,*grid2x,*grid1z,*grid2z;
  int glob_start_z[3];
  int type_ids1,type_ids2;
  double mydiff, diff = 0.0;
  int pdims[2], dim, dim2, mydims[3];
  Plan3D trans_1, trans_2, trans_3, trans_4;
  FILE *fp;

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  //Header
   if(rank == 0) {
	   printf("\n\n[=======================================================================]\n"
			   "[========================P3DFFT Validation test=========================]\n"
			   "[=======================================================================]\n\n");
	   printf("Running on %d cores\n",size);
	   printf("1D FFT in Double precision\n(%d %d %d) grid\n\n",nx,ny,nz);
   }

   // Broadcast input parameters
   MPI_Bcast(&nx,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&ny,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nz,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&dim,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&mem_order,3,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&mem_order2,3,MPI_INT,0,MPI_COMM_WORLD);

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
      printf("Using processor grid %d x %d\n",pdims[0],pdims[1]);

  //---------------------------------------------- P3DFFT Init -------------------------------------------

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
  dim = 2;										// Transform direction
  pgrid1[0] = pdims[0];
  pgrid1[1] = pdims[1];
  pgrid1[2] = 1;

  if (rank == 0)
      printf("Task per direction (%d,%d,%d)\n", pgrid1[0],pgrid1[1],pgrid1[2]);

  //Initialize initial and final grids, based on the above information
  grid1z = p3dfft_init_grid(gdims,pgrid1,proc_order,mem_order,MPI_COMM_WORLD);
  grid2z = p3dfft_init_grid(gdims,pgrid1,proc_order,mem_order,MPI_COMM_WORLD);
  grid1x = p3dfft_init_grid(gdims,pgrid1,proc_order,mem_order2,MPI_COMM_WORLD);
  grid2x = p3dfft_init_grid(gdims,pgrid1,proc_order,mem_order2,MPI_COMM_WORLD);

  //Planning backward z-trasformation
  trans_1 = p3dfft_plan_1Dtrans(grid1z,grid1x,type_ids1,dim,0);
  //Planning forward z-trasformation
  trans_4 = p3dfft_plan_1Dtrans(grid2x,grid2z,type_ids2,dim,0);

  //Determine local z-grid dimensions, so the BEFORE & AFTER transform array dimensions.
  ldimsz = grid1z->ldims;
  sizez = ldimsz[0]*ldimsz[1]*ldimsz[2]*2;
  printf("ldimsz: %d,%d,%d\n", ldimsz[0], ldimsz[1], ldimsz[2]);

  for(i=0;i<3;i++)
    mydims[mem_order[i]] = ldimsz[i];

  //Allocate input and output z-arrays
  INz=(double *) malloc(sizeof(double)*sizez);
  OUTz= (double *) malloc(sizeof(double) *sizez);
  FINz=(double *) malloc(sizeof(double) *sizez);

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


  //-------------------------------------------- Memory filling ----------------------------------------
  //Compute the starting indexes for each processor
  for(i=0;i<3;i++) {
	  glob_start_z[i] = grid1z->glob_start[i];
  }
  printf("Global starting indexes (%d,%d,%d) on rank: %d \n", glob_start_z[0], glob_start_z[1], glob_start_z[2],rank);

  //RAND input
  double *p_in;
  p_in = INz;
  for(y=0;y < ldimsz[1];y++)
	  for(z=0;z < ldimsz[2];z++)
		  for(x=0;x < ldimsz[0];x++) {
			  *p_in++ = rand() % 100;
			  *p_in++ = rand() % 100;
		  }

  //Wave input
  //init_wave(INz,gdims,ldimsz,glob_start_z);

  MPI_Barrier(MPI_COMM_WORLD);

/*  if (rank == 0)
	  printf("INPUT DATA: \n");			//TODO implementare un confronto con un puntatore in OUTz per vedere
 	 	 	 	 	 	 	 	 	 	 	//in che ordine vengono inseriti i dati nell'array (stride-??)
  print_res(INz,mydims,glob_start_z,mem_order,mydims[2]);
*/

  //---------------------------------------------- FFT Execution -------------------------------------------
  //Backward z-transform (z->x grid)
  p3dfft_exec_1Dtrans_double(trans_1,INz,INx);
  normalize(INx,(long int) ldimsx[0]*ldimsx[1]*ldimsx[2],1.0/((double) mydims[mem_order2[2]]));

  //Backward x-transform (x->x grid)
  p3dfft_exec_1Dtrans_double(trans_2,INx,CONV);
  normalize(CONV,(long int) ldimsx[0]*ldimsx[1]*ldimsx[2],1.0/((double) mydims[mem_order[0]]));

  //Forward x-transform (x->x grid)
  p3dfft_exec_1Dtrans_double(trans_3,CONV,OUTx);

  //Forward z-transform (x->z grid)
  p3dfft_exec_1Dtrans_double(trans_4,OUTx,OUTz);

/*  if (rank == 0)
            printf("OUTPUT DATA: \n");
          print_res(OUTz,mydims,glob_start_z,mem_order,mydims[2]);
*/
  //---------------------------------------------- Check Results -------------------------------------------
  mydiff = check_res(INz,OUTz,mydims);

  diff = 0.;
  MPI_Reduce(&mydiff,&diff,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  if(rank == 0)
    printf("Max. diff. =%lg\n",diff);

  //Release memory and clean up P3DFFT++ grids and data
  free(INz); free(OUTz); free(FINz); free(INx); free(CONV); free(OUTx);
  p3dfft_free_grid(grid1x);
  p3dfft_free_grid(grid2x);
  p3dfft_free_grid(grid1z);
  p3dfft_free_grid(grid2z);
  p3dfft_cleanup();

  MPI_Finalize();
}

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
  int imo[3],i,j;
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

void write_buf(double *buf,char *label,int sz[3],int mo[3], int taskid) {
  int i,j,k;
  FILE *fp;
  char str[80],filename[80];
  double *p= buf;

  strcpy(filename,label);
  sprintf(str,".%d",taskid);
  strcat(filename,str);
  fp=fopen(filename,"w");
  for(k=0;k<sz[mo[2]];k++)
    for(j=0;j<sz[mo[1]];j++)
      for(i=0;i<sz[mo[0]];i++) {
	if(abs(*p) > 1.e-7) {
	  fprintf(fp,"(%d %d %d) %lg\n",i,j,k,*p);
	}
	p++;
      }
  fclose(fp); 
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

