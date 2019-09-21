
#include <stdio.h>
#include "Header.h"

/*for a given point, find its cluster by id*/
__device__ cluster_t* findClusterKernel(point_t point, cluster_t* clusters, int k)
{
	int i;
	for (i = 0; i < k; i++)
		if (point.clusterID == clusters[i].id)
			return &(clusters[i]);
	return 0;
}

/*calculates the distance between two positions*/
__device__ double distanceKernel(position_t pointPos, position_t otherPointPos)
{
	return sqrt(pow((pointPos.x - otherPointPos.x), 2) +
		pow((pointPos.y - otherPointPos.y), 2) + pow((pointPos.z - otherPointPos.z), 2));
}

/*for one point- find the closest cluster and move the point to that cluster*/
__global__ void defineCentersKernel(point_t* points, int n, cluster_t* clusters, int k, int* termination, int numOfThreads)
{
	//each thread works on one point
	int i, minIndex;
	cluster_t* currentCluster, *minCluster;
	position_t minCenter;
	double minDistance, currentDistance;
	point_t* point;
	int pointIdx = numOfThreads*blockIdx.x + threadIdx.x;

	if (pointIdx < n)
	{
		point = &(points[pointIdx]);
		currentCluster = findClusterKernel(*point, clusters, k);
		minDistance = distanceKernel(point->pos, currentCluster->center);
		for (i = 0; i < k; i++)
		{
			currentDistance = distanceKernel(point->pos, clusters[i].center);
			if (currentDistance <= minDistance)
			{
				minDistance = currentDistance;
				minIndex = i;
			}
		}

		minCluster = &(clusters[minIndex]);
		minCenter = minCluster->center;

		if (point->clusterID != minCluster->id)
		{
			*termination = 0;
			point->clusterID = minCluster->id;
		}
	}
}

/*use the formula x = x0 + t*v to calculate (x, y, z)*/
__device__ void posAtGivenTimeKernel(point_t* point, double time)
{
	position_t pos;
	pos.x = (point->initialPos.x) + time * (point->v.vx);
	pos.y = (point->initialPos.y) + time * (point->v.vy);
	pos.z = (point->initialPos.z) + time * (point->v.vz);

	point->pos = pos;
}

/*calculate the new position of a point*/
__global__ void updatePointsInTimeKernel(point_t* points, int pointsLength, double time, int numOfThreads)
{
	int pointIdx = numOfThreads*blockIdx.x + threadIdx.x;
	posAtGivenTimeKernel(&points[pointIdx], time);
}

cudaError_t defineCentersWithCuda(point_t* points, int pointsLength, cluster_t* clusters, int clustersLength, int* termination)
{
	point_t* dev_points = 0;
	cluster_t* dev_clusters = 0;
	int* dev_termination = 0;
	cudaError_t cudaStatus;

	/*numOfBlocks idea was taken from NVIDIA  https://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf, page 44*/
	int numOfBlocks = (pointsLength + NUM_OF_THREADS - 1) / NUM_OF_THREADS;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		error(dev_points, dev_clusters, dev_termination, cudaStatus);
	}

	//allocate memory in cuda
	cudaStatus = cudaMalloc((void**)&dev_points, pointsLength * sizeof(point_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		error(dev_points, dev_clusters, dev_termination, cudaStatus);
	}

	cudaStatus = cudaMalloc((void**)&dev_clusters, clustersLength * sizeof(cluster_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		error(dev_points, dev_clusters, dev_termination, cudaStatus);
	}

	cudaStatus = cudaMalloc((void**)&dev_termination, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		error(dev_points, dev_clusters, dev_termination, cudaStatus);
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_points, points, pointsLength * sizeof(point_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy of points failed!");
		error(dev_points, dev_clusters, dev_termination, cudaStatus);
	}

	cudaStatus = cudaMemcpy(dev_clusters, clusters, clustersLength * sizeof(cluster_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy of clusters failed!");
		error(dev_points, dev_clusters, dev_termination, cudaStatus);
	}

	cudaStatus = cudaMemcpy(dev_termination, termination, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy of termination failed!");
		error(dev_points, dev_clusters, dev_termination, cudaStatus);
	}

	defineCentersKernel << <numOfBlocks, NUM_OF_THREADS >> > (dev_points, pointsLength, dev_clusters, clustersLength, dev_termination, NUM_OF_THREADS);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "defineCentersKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		error(dev_points, dev_clusters, dev_termination, cudaStatus);
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching defineCentersKernel!\n", cudaStatus);
		error(dev_points, dev_clusters, dev_termination, cudaStatus);
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(points, dev_points, pointsLength * sizeof(point_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "2 cudaMemcpy of points failed!");
		error(dev_points, dev_clusters, dev_termination, cudaStatus);
	}

	cudaStatus = cudaMemcpy(clusters, dev_clusters, clustersLength * sizeof(cluster_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "2 cudaMemcpy of clusters failed!");
		error(dev_points, dev_clusters, dev_termination, cudaStatus);
	}

	cudaStatus = cudaMemcpy(termination, dev_termination, sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "2 cudaMemcpy of termination failed!");
		error(dev_points, dev_clusters, dev_termination, cudaStatus);
	}

	cudaFree(dev_points);
	cudaFree(dev_clusters);
	cudaFree(dev_termination);
	return cudaStatus;
}

cudaError_t updatePointsInTimeWithCuda(point_t* points, int pointsLength, double time)
{
	point_t* dev_points = 0;
	int numOfBlocks;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		error2(dev_points, cudaStatus);
	}

	cudaStatus = cudaMalloc((void**)&dev_points, pointsLength * sizeof(point_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc points failed!");
		error2(dev_points, cudaStatus);
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_points, points, pointsLength * sizeof(point_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy of points failed!");
		error2(dev_points, cudaStatus);
	}

	numOfBlocks = (pointsLength + NUM_OF_THREADS - 1) / NUM_OF_THREADS;

	updatePointsInTimeKernel << <numOfBlocks, NUM_OF_THREADS >> > (dev_points, pointsLength, time, NUM_OF_THREADS);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "updatePointsInTimeKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		error2(dev_points, cudaStatus);
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching updatePointsInTimeKernel!\n", cudaStatus);
		error2(dev_points, cudaStatus);
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(points, dev_points, pointsLength * sizeof(point_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		error2(dev_points, cudaStatus);
	}

	cudaFree(dev_points);

	return cudaStatus;

}

cudaError_t error(point_t* dev_points, cluster_t* dev_clusters, int* dev_termination, cudaError_t cudaStatus)
{
	cudaFree(dev_points);
	cudaFree(dev_clusters);
	cudaFree(dev_termination);

	return cudaStatus;
}

cudaError_t error2(point_t* dev_points, cudaError_t cudaStatus)
{
	cudaFree(dev_points);

	return cudaStatus;
}
