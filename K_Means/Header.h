#pragma once
#define _CRT_SECURE_NO_WARNINGS

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#define MASTER 0
#define BUFFER_SIZE 100
#define MIN_NUM_OF_POINTS 10000
#define MAX_NUM_OF_POINTS 3000000
#define NUM_OF_THREADS 512
#define INPUT_FILE "C:\\Users\\DELL\\Documents\\Visual Studio 2015\\Projects\\K_Means\\INPUT_FILE2.txt"
#define OUTPUT_FILE "C:\\Users\\DELL\\Documents\\Visual Studio 2015\\Projects\\K_Means\\OUTPUT_FILE2.txt"
struct Position
{
	double x, y, z;
} typedef position_t;

struct Velocity
{
	double vx, vy, vz;
} typedef velocity_t;

struct Point
{
	position_t pos;
	position_t initialPos;
	velocity_t v;
	int clusterID;
} typedef point_t;

struct Cluster
{
	int id;
	position_t center;
	int numOfPoints;
} typedef cluster_t;

/*File related functions*/
FILE* openFileToRead(const char* filePath);
void readAllFile(point_t** points, int* n, int* k, double* t, double* dt, double* limit, double* qm);
void readFirstLine(FILE* file, int* n, int* k, double* t, double* dt, double* limit, double* qm);
void readPoints(FILE* file, int n, point_t* points);
void writeToFile(const char* filePath, double t, double quality, cluster_t* clusters, int k);

/*k-means*/
void KMeans(double* currentTime, double t, double dt, double limit, double qm, double* quality, point_t* points, int n, cluster_t* clusters,
	int k, int jobLen, int rank, int numprocs, MPI_Status* status);
void calcTotalTermination(int* termination, int numprocs, int rank);
void updatePointsInTime(point_t* points, int n, double time);
void posAtGivenTime(point_t* p, double time);
double distance(position_t point, position_t other);
cluster_t* findCluster(point_t point, cluster_t* clusters, int k);
double evaluateQuality(point_t* points, int n, cluster_t* clusters, int k);
double findDiameter(point_t* points, int n, cluster_t* cluster);
void gatherSumsFromSlaves(point_t* points, int n, cluster_t* clusters, int k, int rank, int numprocs, int totalNumOfPoints);
void sumAllPointsInCluster(point_t* points, int n, cluster_t cluster, position_t* sum);
position_t calcAverage(point_t* points, int n, cluster_t cluster, position_t* sumOfPoints, int numOfPoints, int numprocs, int totalNumOfPoints);
void gatherFinalPoints(point_t* finalPoints, int n, point_t* pointsFromSlave, int jobLen);
void calcNumOfPointsInCluster(int* numOfPointsFromAllSlaves, int* numOfPoints, int numprocs);
void recalcNumOfPoints(point_t* points, int n, cluster_t* clusters, int k, int rank, int numOfIteration);
void resetNumOfPoints(cluster_t* clusters, int k);


void createInitialCenters(point_t* points, int n, cluster_t* clusters, int k);
void initializeClustersID(cluster_t* clusters, int k, int n);

/*MPI related functions*/
void packFirstLine(int* position, int* n, int* k, double* t, double* dt, double* limit, double* qm, char* buffer);
void unpackFirstLine(int* packSize, int* n, int* k, double* t, double* dt, double* limit, double* qm, char* buffer);
void createNewMPITypes(MPI_Datatype* PointsMPI, MPI_Datatype* ClustersMPI, MPI_Datatype* PositionMPI);
void sendJobsToSlaves(point_t* points, int n, int numprocs);
void recieveJobFromMaster(point_t** points, int rank, int n, int numprocs, int* recievedLen, MPI_Status* status);
void defineLength(int* len, int n, int rank, int numprocs);

/*cuda*/
cudaError_t error(point_t* dev_points, cluster_t* dev_clusters, int* dev_termination, cudaError_t cudaStatus);
cudaError_t error2(point_t* dev_points, cudaError_t cudaStatus);
cudaError_t defineCentersWithCuda(point_t* points, int n, cluster_t* clusters, int k, int* termination);
cudaError_t updatePointsInTimeWithCuda(point_t* points, int pointsLength, double time);