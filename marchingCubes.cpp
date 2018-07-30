#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <math.h>
#include <GL/glew.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <vector_functions.h>

#include <helper_cuda.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_functions.h>
#include <helper_cuda_gl.h>

#include "marchingCubes.h"

#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif



#define _MY_PI 3.14159

#define SKIP_EMPTY_VOXELS 1

bool g_bValidate = false;

extern "C" void
launch_classifyVoxel(dim3 grid, dim3 threads, uint *voxelVerts, uint *voxelOccupied, float *volume,
uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,
float3 voxelSize, float isoValue);

extern "C" void
launch_classifyVectorVoxel(dim3 grid, dim3 threads, uint *voxelVerts, uint *voxelOccupied, float4 *volume,
uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,
float3 voxelSize, float isoValue);

extern "C" void
launch_compactVoxels(dim3 grid, dim3 threads, uint *compactedVoxelArray, uint *voxelOccupied,
uint *voxelOccupiedScan, uint numVoxels);

extern "C" void
launch_generateTriangles(dim3 grid, dim3 threads,
float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned,
uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
float3 voxelSize, float isoValue, uint activeVoxels, uint maxVerts);

extern "C" void
launch_generateTriangles2(dim3 grid, dim3 threads,
float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned, float *volume,
uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
float3 voxelSize, float isoValue, uint activeVoxels, uint maxVerts);

extern "C" void
launch_generateTriangles3(dim3 grid, dim3 threads,
                         float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned,  float4 *volume,
                         uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
                         float3 voxelSize, float isoValue, uint activeVoxels, uint maxVerts);

extern "C" void allocateTextures(uint **d_edgeTable, uint **d_triTable, uint **d_numVertsTable);
extern "C" void bindVolumeTexture(uchar *d_volume);
extern "C" void ThrustScanWrapper(unsigned int *output, unsigned int *input, unsigned int numElements);


void createVBO(GLuint *vbo, unsigned int size)
{
    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glutReportErrors();
}


void deleteVBO(GLuint *vbo, struct cudaGraphicsResource **cuda_resource)
{
    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);
    //DEPRECATED: checkCudaErrors(cudaGLUnregisterBufferObject(*vbo));
    cudaGraphicsUnregisterResource(*cuda_resource);

    *vbo = 0;
}


MarchingCubes::MarchingCubes()
{
	m_gridSize = make_uint3(0,0,0);
	m_bUseOpenGL = false;

	m_gridSizeLog2 = make_uint3(0,0,0);
	m_gridSizeShift = make_uint3(0,0,0);
	m_gridSize = make_uint3(0,0,0);
	m_gridSizeMask = make_uint3(0,0,0);

	m_voxelSize = make_float3(0.0, 0.0, 0.0);
	m_numVoxels = 0;
	m_maxVerts = 0;
	m_activeVoxels = 0;
	m_totalVerts = 0;

	//  GLint  gl_Shader;
	//  struct cudaGraphicsResource *m_cuda_posvbo_resource, *m_cuda_normalvbo_resource; // handles OpenGL-CUDA exchange
	gl_Shader = 0;
	m_cuda_posvbo_resource = NULL;
	m_cuda_normalvbo_resource = NULL;

	m_dPos = NULL;
	m_dNormal = NULL;

	m_cudaPosVBO = NULL;
	m_cudaNormalVBO = NULL;

	m_dVolume = NULL;
	m_dVoxelVerts = NULL;
	m_dVoxelVertsScan  = NULL;
	m_dVoxelOccupied = NULL;
	m_dVoxelOccupiedScan  = NULL;
	m_dCompVoxelArray = NULL;

	  // tables
	m_dNumVertsTable = NULL;
	m_dEdgeTable = NULL;
	m_dTriTable  = NULL;

	m_posVbo = 0;
	m_normalVbo = 0;


	m_isoValue = 10.0f;
	m_dIsoValue = 0.5f;
	m_bInitialized = false;
}


MarchingCubes::~MarchingCubes()
{
  cleanup();
}


void MarchingCubes::computeIsosurface()
{
	int threads = 128;
	dim3 grid(m_numVoxels / threads, 1, 1);

	// get around maximum grid size of 65535 in each dimension
	if (grid.x > 65535)
	{
		grid.y = grid.x / 32768;
		grid.x = 32768;
	}

	// calculate number of vertices need per voxel


	/*
	 * This sets an 8-bit flag on each voxel
	 * The bits are set if a corner is inside or outside
	 * the iso-surface.
	 *
	 * Writes number (above iso-surface) of vertices in each voxel to m_dVoxelVerts
	 * Writes if voxels are occupied to m_dVoxelOccupied
	 *
	 */



#if COMPUTE_FIELD_DERIVATIVE
	launch_classifyVectorVoxel(grid, threads,
	    m_dVoxelVerts, m_dVoxelOccupied, (float4*)m_dVolume,
	    m_gridSize, m_gridSizeShift, m_gridSizeMask,
	    m_numVoxels, m_voxelSize, m_isoValue);
#else
	launch_classifyVoxel(grid, threads,
		m_dVoxelVerts, m_dVoxelOccupied, m_dVolume,
		m_gridSize, m_gridSizeShift, m_gridSizeMask,
		m_numVoxels, m_voxelSize, m_isoValue);
#endif

#if DEBUG_BUFFERS
	printf("voxelVerts:\n");
	dumpBuffer(m_dVoxelVerts, m_numVoxels, sizeof(uint));
#endif
#if SKIP_EMPTY_VOXELS
	// scan voxel occupied array
	ThrustScanWrapper(m_dVoxelOccupiedScan, m_dVoxelOccupied, m_numVoxels);
#if DEBUG_BUFFERS
	printf("voxelOccupiedScan:\n");
	dumpBuffer(d_voxelOccupiedScan, numVoxels, sizeof(uint));
#endif

	// read back values to calculate total number of non-empty voxels
	// since we are using an exclusive scan, the total is the last value of
	// the scan result plus the last value in the input array
	{
		uint lastElement, lastScanElement;
		checkCudaErrors(cudaMemcpy((void *)&lastElement,
			(void *)(m_dVoxelOccupied + m_numVoxels - 1),
			sizeof(uint), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy((void *)&lastScanElement,
			(void *)(m_dVoxelOccupiedScan + m_numVoxels - 1),
			sizeof(uint), cudaMemcpyDeviceToHost));
		m_activeVoxels = lastElement + lastScanElement;
	}

	if (m_activeVoxels == 0)
	{
		// return if there are no full voxels
		m_totalVerts = 0;
		return;
	}

	/*
	 * I'm pretty sure that this creates m_dCompVoxelArray
	 * which is a list of only the voxels that need to be
	 * processed.
	 */

	launch_compactVoxels(grid, threads, m_dCompVoxelArray, m_dVoxelOccupied, m_dVoxelOccupiedScan, m_numVoxels);

	getLastCudaError("compactVoxels failed");

#endif // SKIP_EMPTY_VOXELS
	// scan voxel vertex count array
	ThrustScanWrapper(m_dVoxelVertsScan, m_dVoxelVerts, m_numVoxels);
#if DEBUG_BUFFERS
	printf("voxelVertsScan:\n");
	dumpBuffer(m_dVoxelVertsScan, m_numVoxels, sizeof(uint));
#endif
	// readback total number of vertices
	{
		uint lastElement, lastScanElement;
		checkCudaErrors(cudaMemcpy((void *)&lastElement,
			(void *)(m_dVoxelVerts + m_numVoxels - 1),
			sizeof(uint), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy((void *)&lastScanElement,
			(void *)(m_dVoxelVertsScan + m_numVoxels - 1),
			sizeof(uint), cudaMemcpyDeviceToHost));
		m_totalVerts = lastElement + lastScanElement;
	}
	// generate triangles, writing to vertex buffers
	if (!g_bValidate)
	{
		size_t num_bytes;
		// DEPRECATED: checkCudaErrors(cudaGLMapBufferObject((void**)&d_pos, posVbo));
		checkCudaErrors(cudaGraphicsMapResources(1, &m_cuda_posvbo_resource, 0));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&m_dPos, &num_bytes, m_cuda_posvbo_resource));

		// DEPRECATED: checkCudaErrors(cudaGLMapBufferObject((void**)&d_normal, normalVbo));
		checkCudaErrors(cudaGraphicsMapResources(1, &m_cuda_normalvbo_resource, 0));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&m_dNormal, &num_bytes, m_cuda_normalvbo_resource));
	}

#if SKIP_EMPTY_VOXELS
	dim3 grid2((int)ceil(m_activeVoxels / (float)NTHREADS), 1, 1);
#else
	dim3 grid2((int)ceil(m_numVoxels / (float)NTHREADS), 1, 1);
#endif

	while (grid2.x > 65535)
	{
		grid2.x /= 2;
		grid2.y *= 2;
	}

#if SAMPLE_VOLUME

#if COMPUTE_FIELD_DERIVATIVE
	launch_generateTriangles3(grid2, NTHREADS,  m_dPos, m_dNormal,
	    m_dCompVoxelArray,
	    m_dVoxelVertsScan,(float4*)m_dVolume,
	    m_gridSize, m_gridSizeShift, m_gridSizeMask,
	    m_voxelSize, m_isoValue, m_activeVoxels,
	    m_maxVerts);
#else
	launch_generateTriangles2(grid2, NTHREADS, m_dPos, m_dNormal,
		m_dCompVoxelArray,
		m_dVoxelVertsScan, m_dVolume,
		m_gridSize, m_gridSizeShift, m_gridSizeMask,
		m_voxelSize, m_isoValue, m_activeVoxels,
		m_maxVerts);
#endif

#else
	launch_generateTriangles(grid2, NTHREADS, m_dPos, m_dNormal,
		m_dCompVoxelArray,
		m_dVoxelVertsScan,
		m_gridSize, m_gridSizeShift, m_gridSizeMask,
		m_voxelSize, m_isoValue, m_activeVoxels,
		m_maxVerts);
#endif

	if (!g_bValidate)
	{
		// DEPRECATED:      checkCudaErrors(cudaGLUnmapBufferObject(normalVbo));
		checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cuda_normalvbo_resource, 0));
		// DEPRECATED:      checkCudaErrors(cudaGLUnmapBufferObject(posVbo));
		checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cuda_posvbo_resource, 0));
	}

}


void MarchingCubes::renderIsosurface()
{
	glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_normalVbo);
	glNormalPointer(GL_FLOAT, sizeof(float) * 4, 0);
	glEnableClientState(GL_NORMAL_ARRAY);

	glColor3f(0.0, 1.0, 0.0);
	glDrawArrays(GL_TRIANGLES, 0, m_totalVerts);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

    /*
     * Enable the code below to dump raw image data to file
     */

	/*
	static int render_count = 0;
	render_count++;
	if(render_count%100 == 0)
	{
	  fprintf(stderr,"Printing raw file %d", render_count);
      char filename[56];
	  sprintf(filename,"file_%d.raw", render_count);
	  unsigned char* imageData = (unsigned char *)malloc((int)(640*480*(3)));
	  glReadPixels(0, 0, 640, 480, GL_RGB, GL_UNSIGNED_BYTE, imageData);
	  std::ofstream binaryFile (filename, std::ios::out | std::ios::binary);
	  binaryFile.write ((char*)imageData, 640*480*3);
	  binaryFile.close();
	}
	*/

}


void MarchingCubes::init(uint logGridSize, bool bUseOpenGL)
{
	m_gridSizeLog2.x = logGridSize/3;
	m_gridSizeLog2.y = logGridSize/3;
	m_gridSizeLog2.z = logGridSize/3;

	m_gridSize = make_uint3(1<<m_gridSizeLog2.x, 1<<m_gridSizeLog2.y, 1<<m_gridSizeLog2.z);

	m_gridSizeMask = make_uint3(m_gridSize.x - 1, m_gridSize.y - 1, m_gridSize.z - 1);
	m_gridSizeShift = make_uint3(0, m_gridSizeLog2.x, m_gridSizeLog2.x + m_gridSizeLog2.y);

	m_numVoxels = m_gridSize.x*m_gridSize.y*m_gridSize.z;
	m_voxelSize = make_float3(2.0f / m_gridSize.x, 2.0f / m_gridSize.y, 2.0f / m_gridSize.z);
	m_maxVerts = m_gridSize.x*m_gridSize.y * 100;

	printf("Marching cubes init:\n");
    printf("  grid: %d x %d x %d = %d voxels\n", m_gridSize.x, m_gridSize.y, m_gridSize.z, m_numVoxels);
    printf("  max verts = %d\n", m_maxVerts);

	//checkCudaErrors(cudaMalloc((void **)&m_dVolume, size));
	//bindVolumeTexture(m_dVolume);

	if (g_bValidate)
	{
		cudaMalloc((void **)&(m_dPos), m_maxVerts*sizeof(float) * 4);
		cudaMalloc((void **)&(m_dNormal), m_maxVerts*sizeof(float) * 4);
	}
	else
	{
		// create VBOs
		createVBO(&m_posVbo, m_maxVerts*sizeof(float) * 4);
		// DEPRECATED: checkCudaErrors( cudaGLRegisterBufferObject(posVbo) );
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_cuda_posvbo_resource, m_posVbo,
			cudaGraphicsMapFlagsWriteDiscard));

		createVBO(&m_normalVbo, m_maxVerts*sizeof(float) * 4);
		// DEPRECATED: checkCudaErrors(cudaGLRegisterBufferObject(normalVbo));
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_cuda_normalvbo_resource, m_normalVbo,
			cudaGraphicsMapFlagsWriteDiscard));
	}

	// allocate textures
	allocateTextures(&m_dEdgeTable, &m_dTriTable, &m_dNumVertsTable);

	// allocate device memory
	unsigned int memSize = sizeof(uint) * m_numVoxels;
	checkCudaErrors(cudaMalloc((void **)&m_dVoxelVerts, memSize));
	checkCudaErrors(cudaMalloc((void **)&m_dVoxelVertsScan, memSize));
	checkCudaErrors(cudaMalloc((void **)&m_dVoxelOccupied, memSize));
	checkCudaErrors(cudaMalloc((void **)&m_dVoxelOccupiedScan, memSize));
	checkCudaErrors(cudaMalloc((void **)&m_dCompVoxelArray, memSize));
}


void MarchingCubes::cleanup()
{
	if (g_bValidate)
	{
		cudaFree(m_dPos);
		cudaFree(m_dNormal);
	}
	else
	{
		//sdkDeleteTimer(&timer);

		deleteVBO(&m_posVbo, &m_cuda_posvbo_resource);
		deleteVBO(&m_normalVbo, &m_cuda_normalvbo_resource);
	}

	checkCudaErrors(cudaFree(m_dEdgeTable));
	checkCudaErrors(cudaFree(m_dTriTable));
	checkCudaErrors(cudaFree(m_dNumVertsTable));

	checkCudaErrors(cudaFree(m_dVoxelVerts));
	checkCudaErrors(cudaFree(m_dVoxelVertsScan));
	checkCudaErrors(cudaFree(m_dVoxelOccupied));
	checkCudaErrors(cudaFree(m_dVoxelOccupiedScan));
	checkCudaErrors(cudaFree(m_dCompVoxelArray));
}
