/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/*
Marching cubes

This sample extracts a geometric isosurface from a volume dataset using
the marching cubes algorithm. It uses the scan (prefix sum) function from
the Thrust library to perform stream compaction.  Similar techniques can
be used for other problems that require a variable-sized output per
thread.

For more information on marching cubes see:
http://local.wasp.uwa.edu.au/~pbourke/geometry/polygonise/
http://en.wikipedia.org/wiki/Marching_cubes

Volume data courtesy:
http://www9.informatik.uni-erlangen.de/External/vollib/

For more information on the Thrust library
http://code.google.com/p/thrust/

The algorithm consists of several stages:

1. Execute "classifyVoxel" kernel
This evaluates the volume at the corners of each voxel and computes the
number of vertices each voxel will generate.
It is executed using one thread per voxel.
It writes two arrays - voxelOccupied and voxelVertices to global memory.
voxelOccupied is a flag indicating if the voxel is non-empty.

2. Scan "voxelOccupied" array (using Thrust scan)
Read back the total number of occupied voxels from GPU to CPU.
This is the sum of the last value of the exclusive scan and the last
input value.

3. Execute "compactVoxels" kernel
This compacts the voxelOccupied array to get rid of empty voxels.
This allows us to run the complex "generateTriangles" kernel on only
the occupied voxels.

4. Scan voxelVertices array
This gives the start address for the vertex data for each voxel.
We read back the total number of vertices generated from GPU to CPU.

Note that by using a custom scan function we could combine the above two
scan operations above into a single operation.

5. Execute "generateTriangles" kernel
This runs only on the occupied voxels.
It looks up the field values again and generates the triangle data,
using the results of the scan to write the output to the correct addresses.
The marching cubes look-up tables are stored in 1D textures.

6. Render geometry
Using number of vertices from readback.
*/

#ifndef __MARCHING_CUBES_H__
#define __MARCHING_CUBES_H__

// includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <GL/glew.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <vector_functions.h>

#include <helper_cuda.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_functions.h>
#include <helper_cuda_gl.h>

#include "defines.h"

#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <helper_functions.h>
#include "particles_kernel.cuh"
#include "vector_functions.h"


class MarchingCubes
{
public:

	  MarchingCubes();
	  ~MarchingCubes();

	  void cleanup();

	  void init(uint logGridSize, bool bUseOpenGL);

	  void computeIsosurface();

	  void renderIsosurface();

	  float *getVolume() const
	  {
		  return m_dVolume;
	  }

	  void setVolume(float *vol)
	  {
		  m_dVolume = vol;
	  }

	  unsigned int getCurrentReadBuffer() const
	  {
		  return m_posVbo;
	  }

	  unsigned int getNormalBuffer()       const
	  {
		  return m_normalVbo;
	  }

	  void *getCudaPosVBO()              const
	  {
		  return (void *)m_cudaPosVBO;
	  }

	  void *getCudaNormalVBO()            const
	  {
		  return (void *)m_cudaNormalVBO;
	  }

	  float m_isoValue;
	  float m_dIsoValue;


protected: // data

	  bool m_bUseOpenGL;
	  bool m_bInitialized;

	  uint3 m_gridSizeLog2;
	  uint3 m_gridSizeShift;
	  uint3 m_gridSize;
	  uint3 m_gridSizeMask;

	  float3 m_voxelSize;
	  uint m_numVoxels;
	  uint m_maxVerts;
	  uint m_activeVoxels;
	  uint m_totalVerts;



	  GLuint m_posVbo;            // vertex buffer object for particle positions
	  GLuint m_normalVbo;         // vertex buffer object for colors
	  float *m_cudaPosVBO;        // these are the CUDA deviceMem Pos
	  float *m_cudaNormalVBO;     // these are the CUDA deviceMem Color
	  //GLuint m_cudaPosVBO;
	  //GLuint m_cudaNormalVBO;

	  // device data

	  GLint  gl_Shader;
	  struct cudaGraphicsResource *m_cuda_posvbo_resource;
	  struct cudaGraphicsResource *m_cuda_normalvbo_resource; // handles OpenGL-CUDA exchange

	  float4 *m_dPos;
	  float4 *m_dNormal;

	  float *m_dVolume;
	  uint  *m_dVoxelVerts;
	  uint  *m_dVoxelVertsScan;
	  uint  *m_dVoxelOccupied;
	  uint  *m_dVoxelOccupiedScan;
	  uint  *m_dCompVoxelArray;

	  // tables
	  uint *m_dNumVertsTable;
	  uint *m_dEdgeTable;
	  uint *m_dTriTable;
};

#endif // __MARCHING_CUBES_H__

