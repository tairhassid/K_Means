#include "Header.h"

MPI_Datatype PointsMPI;
MPI_Datatype ClustersMPI;
MPI_Datatype PositionMPI;
int main(int argc, char *argv[])
{
	int n, k;
	double limit, qm, t, dt, currentTime = 0, quality = 0;
	double startTime, endTime, totalRunningTime;
	point_t* points;
	cluster_t* clusters;

	//MPI variables
	int numprocs, rank, lengthToSend = 0, jobLen = 0;
	MPI_Status status;
	char buffer[BUFFER_SIZE];
	int position = 0;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (numprocs < 3) {
		printf("Please run at least three MPI proceeses\n");
		fflush(stdout);
		MPI_Abort(MPI_COMM_WORLD, 0);
	}

	if (rank == MASTER)
	{
		startTime = MPI_Wtime();
		readAllFile(&points, &n, &k, &t, &dt, &limit, &qm);
		
		packFirstLine(&position, &n, &k, &t, &dt, &limit, &qm, buffer);
	}

	MPI_Bcast(&position, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(buffer, position, MPI_PACKED, 0, MPI_COMM_WORLD);
	unpackFirstLine(&position, &n, &k, &t, &dt, &limit, &qm, buffer);

	clusters = (cluster_t*)calloc(k, sizeof(cluster_t));
	createNewMPITypes(&PointsMPI, &ClustersMPI, &PositionMPI);

	if (rank == MASTER)
	{
		createInitialCenters(points, n, clusters, k);
		initializeClustersID(clusters, k, n);

		sendJobsToSlaves(points, n, numprocs);
		defineLength(&jobLen, n, rank, numprocs);

	}
	else
		recieveJobFromMaster(&points, rank, n, numprocs, &jobLen, &status);
	

	MPI_Bcast(clusters, k, ClustersMPI, 0, MPI_COMM_WORLD);
	clusters[0].numOfPoints = jobLen;

	KMeans(&currentTime, t, dt, limit, qm, &quality, points, n, clusters, k, jobLen, rank, numprocs, &status);

	if (rank == MASTER)
	{
		if (quality >= qm)
			currentTime -= dt; //because after the last iteration dt is added once more and not entering the loop
		writeToFile(OUTPUT_FILE, currentTime, quality, clusters, k);

		endTime = MPI_Wtime();
		totalRunningTime = endTime - startTime;
		printf("total running time = %f\n", totalRunningTime);
		fflush(stdout);
	}

	free(clusters);
	free(points);

	printf("proc %d finished\n", rank);
	fflush(stdout);
	MPI_Finalize();
	return 0;
}

FILE* openFileToRead(const char* filePath)
{
	FILE* file = fopen(filePath, "r");
	if (file == NULL)
	{
		printf("Couldn't find the file\n");
		fflush(stdout);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	return file;
}

void readAllFile(point_t** points, int* n, int* k, double* t, double* dt, double* limit, double* qm)
{
	FILE* file;
	file = openFileToRead(INPUT_FILE);
	readFirstLine(file, n, k, t, dt, limit, qm);

	*points = (point_t*)calloc(*n, sizeof(point_t));

	readPoints(file, *n, *points);
}

void readFirstLine(FILE* file, int* n, int* k, double* t, double* dt, double* limit, double* qm)
{
	fscanf(file, "%d %d %lf %lf %lf %lf", n, k, t, dt, limit, qm);

	if (*n < MIN_NUM_OF_POINTS || *n > MAX_NUM_OF_POINTS)
	{
		printf("The number of points has to be between %d and %d", MIN_NUM_OF_POINTS, MAX_NUM_OF_POINTS);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
}

void readPoints(FILE* file, int n, point_t* points)
{
	int i;

	for (i = 0; i < n; i++)
	{
		fscanf(file, "%lf %lf %lf %lf %lf %lf", &points[i].pos.x, &points[i].pos.y, &points[i].pos.z,
			&points[i].v.vx, &points[i].v.vy, &points[i].v.vz);
		
		points[i].initialPos.x = points[i].pos.x;
		points[i].initialPos.y = points[i].pos.y;
		points[i].initialPos.z = points[i].pos.z;
	}
	fclose(file);
}

void writeToFile(const char* filePath, double t, double quality, cluster_t* clusters, int k)
{
	int i;
	FILE* file = fopen(filePath, "w");
	if (file == NULL)
	{
		printf("Failed opening the file. Exiting!\n");
		fflush(stdout);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	fprintf(file, "First occurrence t = %lf  with q = %lf\n", t, quality);
	fprintf(file, "Centers of the clusters:\n");

	for (i = 0; i < k; i++)
	{
		fprintf(file, "%lf  %lf  %lf\n", clusters[i].center.x, clusters[i].center.y, clusters[i].center.z);
	}
	printf("IN WRITE\n");
	fflush(stdout);

	fclose(file);
}

void KMeans(double* currentTime, double t, double dt, double limit, double qm, double* quality, point_t* points, int n, cluster_t* clusters,
	int k, int jobLen, int rank, int numprocs, MPI_Status* status)
{
	/*termination- indicates if there was at least one point that moved from one cluster to another*/
	int termination;
	int numOfIteration, i;

	for (*currentTime = 0; *currentTime <= t; *currentTime += dt)
	{
		updatePointsInTimeWithCuda(points, jobLen, *currentTime);

		termination = 0;
		numOfIteration = 0;
		while (!termination && numOfIteration < limit)
		{
			numOfIteration++;

			termination = 1;
			defineCentersWithCuda(points, jobLen, clusters, k, &termination);
			recalcNumOfPoints(points, jobLen, clusters, k, rank, numOfIteration);

			gatherSumsFromSlaves(points, jobLen, clusters, k, rank, numprocs, n);
			calcTotalTermination(&termination, numprocs, rank);
		}

		if (rank == MASTER)
		{
			int add = 0;
			for (i = 1; i < numprocs; i++)
			{
				int slaveJobLen;
				defineLength(&slaveJobLen, n, i, numprocs);
				MPI_Recv(&points[jobLen*i + add], slaveJobLen, PointsMPI, i, 0, MPI_COMM_WORLD, status);
				add = slaveJobLen - jobLen;
			}
			*quality = evaluateQuality(points, n, clusters, k);
		}
		else
		{
			MPI_Send(points, jobLen, PointsMPI, MASTER, 0, MPI_COMM_WORLD);
		}
		MPI_Bcast(quality, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
		if (*quality < qm)
			break;
	}
}

void calcTotalTermination(int* termination, int numprocs, int rank)
{
	int i;
	int* terminationFromAllProcs = (int*)calloc(numprocs, sizeof(int));

	MPI_Gather(termination, 1, MPI_INT, terminationFromAllProcs, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	if (rank == MASTER)
		for (i = 0; i < numprocs; i++)
			if (terminationFromAllProcs[i] == 0)
				*termination = 0;

	MPI_Bcast(termination, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	free(terminationFromAllProcs);
}

/*update all the points*/
void updatePointsInTime(point_t* points, int n, double time)
{
	int i;
#pragma omp parallel for private(i)
	for (i = 0; i < n; i++)
	{
		posAtGivenTime(&points[i], time);
	}
}

/*changes the position of a given point according to the time. x(t) = x(t-1) + t*v  */
void posAtGivenTime(point_t* p, double time)
{
	position_t pos;
	pos.x = (p->pos.x) + time * (p->v.vx);
	pos.y = (p->pos.y) + time * (p->v.vy);
	pos.z = (p->pos.z) + time * (p->v.vz);

	p->pos = pos;
	//return pos;
}

/*first k points are the initial centers*/
void createInitialCenters(point_t* points, int n, cluster_t* clusters, int k)
{
	int i;
#pragma omp parallel for private(i)
	for (i = 0; i < k; i++)
	{
		clusters[i].center.x = points[i].pos.x;
		clusters[i].center.y = points[i].pos.y;
		clusters[i].center.z = points[i].pos.z;
	}

}

/*each cluster gets id equals to its location in the array*/
void initializeClustersID(cluster_t* clusters, int k, int n)
{
	int i;
#pragma omp parallel for private(i)
	for (i = 0; i < k; i++)
	{
		clusters[i].id = i;
	}
}

/*Counts the points for every cluster*/
void recalcNumOfPoints(point_t* points, int n, cluster_t* clusters, int k, int rank, int numOfIteration)
{
	int i;

	resetNumOfPoints(clusters, k);

	for (i = 0; i < n; i++)
		clusters[points[i].clusterID].numOfPoints++;
}

/*initialize numOfPoints for each cluster to 0*/
void resetNumOfPoints(cluster_t* clusters, int k)
{
	int i;
#pragma omp parallel private(i)
	{
		#pragma omp for
		for (i = 0; i < k; i++)
			clusters[i].numOfPoints = 0;
	}
}

/*calculate distance between two points*/
double distance(position_t pointPos, position_t otherPointPos)
{
	return sqrt(pow((pointPos.x - otherPointPos.x), 2) +
		pow((pointPos.y - otherPointPos.y), 2) + pow((pointPos.z - otherPointPos.z), 2));
}

/*Gets a point and returns the cluster it belongs*/
cluster_t* findCluster(point_t point, cluster_t* clusters, int k)
{
	int i;
	for (i = 0; i < k; i++)
		if (point.clusterID == clusters[i].id)
			return &(clusters[i]);
	return &(clusters[0]);
}

/*Gather sums of all points and num of points from each slave. Calculates new clusters (average of all points in each)*/
void gatherSumsFromSlaves(point_t* points, int jobLen, cluster_t* clusters, int k, int rank, int numprocs, int totalNumOfPoints)
{
	int i, *numOfPointsFromAllSlaves;
	int numOfPointsInCluster;
	position_t sum = { 0, 0, 0 };
	position_t* sumFromAllSlaves;
	position_t average;

	numOfPointsFromAllSlaves = (int*)malloc(numprocs * sizeof(int));
	sumFromAllSlaves = (position_t*)malloc(numprocs * sizeof(position_t));

	for (i = 0; i < k; i++)
	{
		numOfPointsInCluster = 0;

		MPI_Gather(&(clusters[i].numOfPoints), 1, MPI_INT, numOfPointsFromAllSlaves, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
		if (rank == MASTER)
			calcNumOfPointsInCluster(numOfPointsFromAllSlaves, &numOfPointsInCluster, numprocs);
		MPI_Bcast(&numOfPointsInCluster, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

		if (numOfPointsInCluster > 0)
		{
			sumAllPointsInCluster(points, jobLen, clusters[i], &sum);

			MPI_Gather(&sum, 1, PositionMPI, sumFromAllSlaves, 1, PositionMPI, MASTER, MPI_COMM_WORLD);
			if (rank == MASTER)
			{
				average = calcAverage(points, jobLen, clusters[i], sumFromAllSlaves, numOfPointsInCluster, numprocs, totalNumOfPoints);
			}
			MPI_Bcast(&average, 1, PositionMPI, MASTER, MPI_COMM_WORLD);

			clusters[i].center = average;
		}

	}
	free(sumFromAllSlaves);
	free(numOfPointsFromAllSlaves);
}

/*Gets an array in which every item is numOfPoints in a specific cluster, that the master got from the slaves*/
void calcNumOfPointsInCluster(int* numOfPointsFromAllSlaves, int* numOfPoints, int numprocs)
{
	int i;
	for (i = 0; i < numprocs; i++)
		*numOfPoints += numOfPointsFromAllSlaves[i];
}

/*Sums x, y, z for all points in given cluster*/
void sumAllPointsInCluster(point_t* points, int n, cluster_t cluster, position_t* sum)
{
	int i;
	sum->x = sum->y = sum->z = 0;
	for (i = 0; i < n; i++)
		if (cluster.id == points[i].clusterID)
		{
			sum->x += points[i].pos.x;
			sum->y += points[i].pos.y;
			sum->z += points[i].pos.z;
		}
}

/*Calculates the average of a given cluster*/
position_t calcAverage(point_t* points, int n, cluster_t cluster, position_t* sumOfPoints, int numOfPoints, int numprocs, int totalNumOfPoints)
{
	int i;
	position_t totalSumForCluster = { 0, 0, 0 };
	position_t average = { 0, 0, 0 };
	for (i = 0; i < numprocs; i++)
	{
		totalSumForCluster.x += sumOfPoints[i].x;
		totalSumForCluster.y += sumOfPoints[i].y;
		totalSumForCluster.z += sumOfPoints[i].z;
	}

	if (numOfPoints > 0)
	{
		average.x = totalSumForCluster.x / numOfPoints;
		average.y = totalSumForCluster.y / numOfPoints;
		average.z = totalSumForCluster.z / numOfPoints;
	}

	return average;
}

double evaluateQuality(point_t* points, int n, cluster_t* clusters, int k)
{
	int i, j;
	double q = 0.0;
	double diameter, distanceBewteenClusters;

#pragma omp parallel for private(i, j, distanceBewteenClusters) reduction(+:q)
	for (i = 0; i < k; i++)
	{
		diameter = findDiameter(points, n, &(clusters[i]));
		for (j = 0; j < k; j++)
		{
			if (j != i)
			{
				distanceBewteenClusters = distance(clusters[i].center, clusters[j].center);
				q += diameter / distanceBewteenClusters;
			}
		}
	}
	return q / (k*(k - 1));
}

double findDiameter(point_t* points, int n, cluster_t* cluster)
{
	int i, j;
	double maxDist = 0.0;
	double currentDist;

#pragma omp parallel for private(i, j, currentDist)
	for (i = 0; i < n; i++)
		if (points[i].clusterID == cluster->id)
		{
			for (j = i+1; j < n; j++)
			{
				if (points[j].clusterID == cluster->id)
				{
					currentDist = distance(points[i].pos, points[j].pos);
					if (currentDist > maxDist)
						maxDist = currentDist;
				}
			}
		}
	
	return maxDist;
}

/*packs the first line read from the file in order to send it to the other processes.
position - how long the whole package is*/
void packFirstLine(int* position, int* n, int* k, double* t, double* dt, double* limit, double* qm, char* buffer)
{
	MPI_Pack(n, 1, MPI_INT, buffer, BUFFER_SIZE, position, MPI_COMM_WORLD);
	MPI_Pack(k, 1, MPI_INT, buffer, BUFFER_SIZE, position, MPI_COMM_WORLD);
	MPI_Pack(t, 1, MPI_DOUBLE, buffer, BUFFER_SIZE, position, MPI_COMM_WORLD);
	MPI_Pack(dt, 1, MPI_DOUBLE, buffer, BUFFER_SIZE, position, MPI_COMM_WORLD);
	MPI_Pack(limit, 1, MPI_DOUBLE, buffer, BUFFER_SIZE, position, MPI_COMM_WORLD);
	MPI_Pack(qm, 1, MPI_DOUBLE, buffer, BUFFER_SIZE, position, MPI_COMM_WORLD);
}

/*unpacks the package sent from the master*/
void unpackFirstLine(int* packSize, int* n, int* k, double* t, double* dt, double* limit, double* qm, char* buffer)
{
	int position = 0;
	MPI_Unpack(buffer, *packSize, &position, n, 1, MPI_INT, MPI_COMM_WORLD);
	MPI_Unpack(buffer, *packSize, &position, k, 1, MPI_INT, MPI_COMM_WORLD);
	MPI_Unpack(buffer, *packSize, &position, t, 1, MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Unpack(buffer, *packSize, &position, dt, 1, MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Unpack(buffer, *packSize, &position, limit, 1, MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Unpack(buffer, *packSize, &position, qm, 1, MPI_DOUBLE, MPI_COMM_WORLD);
}

void createNewMPITypes(MPI_Datatype* PointsMPI, MPI_Datatype* ClustersMPI, MPI_Datatype* PositionMPI)
{
	position_t position;
	MPI_Datatype posType[3] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
	int posBlockLen[3] = { 1, 1, 1 };
	MPI_Aint posDisp[3];

	velocity_t velocity;
	MPI_Datatype VelocityMPI;
	MPI_Datatype velocityType[3] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
	int velocityBlockLen[3] = { 1, 1, 1 };
	MPI_Aint velocityDisp[3];

	posDisp[0] = (char*)&position.x - (char*)&position;
	posDisp[1] = (char*)&position.y - (char*)&position;
	posDisp[2] = (char*)&position.z - (char*)&position;

	MPI_Type_create_struct(3, posBlockLen, posDisp, posType, PositionMPI);
	MPI_Type_commit(PositionMPI);

	velocityDisp[0] = (char*)&velocity.vx - (char*)&velocity;
	velocityDisp[1] = (char*)&velocity.vy - (char*)&velocity;
	velocityDisp[2] = (char*)&velocity.vz - (char*)&velocity;

	MPI_Type_create_struct(3, velocityBlockLen, velocityDisp, velocityType, &VelocityMPI);
	MPI_Type_commit(&VelocityMPI);

	point_t point;
	MPI_Datatype pointsType[4] = { *PositionMPI, *PositionMPI, VelocityMPI, MPI_INT };
	int pointsBlockLen[4] = { 1, 1, 1, 1 };
	MPI_Aint pointsDisp[4];

	pointsDisp[0] = (char*)&point.pos - (char*)&point;
	pointsDisp[1] = (char*)&point.initialPos - (char*)&point;
	pointsDisp[2] = (char*)&point.v - (char*)&point;
	pointsDisp[3] = (char*)&point.clusterID - (char*)&point;

	MPI_Type_create_struct(4, pointsBlockLen, pointsDisp, pointsType, PointsMPI);
	MPI_Type_commit(PointsMPI);

	cluster_t cluster;
	MPI_Datatype clustersType[3] = { MPI_INT, *PositionMPI, MPI_INT };
	int clustersBlockLen[3] = { 1, 1, 1 };
	MPI_Aint clustersDisp[3];

	clustersDisp[0] = (char*)&cluster.id - (char*)&cluster;
	clustersDisp[1] = (char*)&cluster.center - (char*)&cluster;
	clustersDisp[2] = (char*)&cluster.numOfPoints - (char*)&cluster;

	MPI_Type_create_struct(3, clustersBlockLen, clustersDisp, clustersType, ClustersMPI);
	MPI_Type_commit(ClustersMPI);
}

/*Sends the relevant part of the points-array to each process to work on*/
void sendJobsToSlaves(point_t* points, int n, int numprocs)
{
	point_t* p = points;
	int i, lengthToSend = 0;
	for (i = 0; i < numprocs; i++)
	{
		defineLength(&lengthToSend, n, i, numprocs);

		if (i != 0)
			MPI_Send(p, lengthToSend, PointsMPI, i, 0, MPI_COMM_WORLD);
		p += lengthToSend;
	}
}

void recieveJobFromMaster(point_t** points, int rank, int n, int numprocs, int* recievedLen, MPI_Status* status)
{

	defineLength(recievedLen, n, rank, numprocs);

	*points = (point_t*)calloc(*recievedLen, sizeof(point_t));

	MPI_Recv(*points, *recievedLen, PointsMPI, MASTER, 0, MPI_COMM_WORLD, status);
}

/*Divides the number of points to the processes, if n does not devide by numprocs equaly, each process gets one
extra point of the remainder*/
void defineLength(int* len, int n, int rank, int numprocs)
{
	if (rank < n%numprocs)
		*len = n / numprocs + 1;
	else
		*len = n / numprocs;
}

void gatherFinalPoints(point_t* finalPoints, int n, point_t* pointsFromSlave, int jobLen)
{
	MPI_Gather(pointsFromSlave, jobLen, PointsMPI, pointsFromSlave, jobLen, PointsMPI, MASTER, MPI_COMM_WORLD);
}


