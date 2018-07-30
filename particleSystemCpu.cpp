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

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <GL/glew.h>

#include "particleSystemCpu.h"


#include "defines.h"

#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif


// Simple helpers to move data between float4, float3 and float arrays
// Careful though these are DANGEROUS...

float4 getFloat4(float* f, uint idx)
{
	return make_float4(f[4*idx], f[4*idx+1], f[4*idx+2], 0.0f );
}

float3 getFloat3(float* f, uint idx)
{
	return make_float3(f[4*idx], f[4*idx+1], f[4*idx+2] );
}

void setFloat(float* f, uint idx, float4 f4)
{
	f[4*idx]     = f4.x;
	f[4*idx + 1] = f4.y;
	f[4*idx + 2] = f4.z;
	f[4*idx + 3] = f4.w;
}

void setFloat(float* f, uint idx, float3 f3)
{
	f[4*idx]     = f3.x;
	f[4*idx + 1] = f3.y;
	f[4*idx + 2] = f3.z;
	f[4*idx + 3] = 0.0f;
}


ParticleSystemCpu::ParticleSystemCpu(uint numParticles, uint3 gridSize, bool bUseOpenGL) : ParticleSystem(numParticles, gridSize, bUseOpenGL)
{
    m_bInitialized = false;
    m_hPos=0;
    m_hVel=0;
    m_timer=NULL;
    m_solverIterations=1;

    m_numGridCells = m_gridSize.x*m_gridSize.y*m_gridSize.z;
    //    float3 worldSize = make_float3(2.0f, 2.0f, 2.0f);

    m_gridSortBits = 18;    // increase this for larger grids

    // set simulation parameters
    m_params.gridSize = m_gridSize;
    m_params.gridLogSize = make_uint3(
    		floor(log2((float)m_gridSize.x)),
    		floor(log2((float)m_gridSize.y)),
    		floor(log2((float)m_gridSize.z)) );

    fprintf(stderr,"ParticleSystemCpu Grid Log size %u x %u %u\n",
    		m_params.gridLogSize.x, m_params.gridLogSize.y, m_params.gridLogSize.z);

    m_params.numCells = m_numGridCells;
    m_params.numBodies = m_numParticles;

    m_params.particleRadius = 1.0f / 64.0f;
    m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
    m_params.colliderRadius = 0.2f;

    m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
    //    m_params.cellSize = make_float3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z / m_gridSize.z);
    float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
    m_params.cellSize = make_float3(cellSize, cellSize, cellSize);

    m_params.spring = 0.5f;
    m_params.damping = 0.02f;
    m_params.shear = 0.1f;
    m_params.attraction = 0.0f;
    m_params.boundaryDamping = -0.5f;
    m_params.gravity = make_float3(0.0f, -0.0003f, 0.0f);
    m_params.globalDamping = 1.0f;

    _initialize(numParticles);
}


ParticleSystemCpu::~ParticleSystemCpu()
{
	fprintf(stderr,"Calling ParticleSystemCpu destructor!\n");
    _finalize();
    m_numParticles = 0;
}


uint ParticleSystemCpu::createVBO(uint size)
{
    return 0;
}


void ParticleSystemCpu::_initialize(int numParticles)
{
    // allocate CPU data
    unsigned int memSize = sizeof(float) * 4 * m_numParticles;

    assert(!m_bInitialized);

    m_numParticles = numParticles;

	printf("ParticleSystemCpu _initialize:\n");
	printf("  Number of particles:%u \n", m_numParticles);
	printf("  Grid dimensios: %u x %u x%u\n", m_gridSize.x, m_gridSize.y, m_gridSize.z);
	printf("  Number of grid cells: %u \n", m_numGridCells);

	// allocate host storage for field data
	m_hScalarField = new float[m_numGridCells];
	memset(m_hScalarField, 0, m_numGridCells*sizeof(float));

    // allocate host storage
    m_hPos = new float[m_numParticles*4];
    m_hVel = new float[m_numParticles*4];
    memset(m_hPos, 0, m_numParticles*4*sizeof(float));
    memset(m_hVel, 0, m_numParticles*4*sizeof(float));

    m_hCellStart = new uint[m_numGridCells];
    memset(m_hCellStart, 0, m_numGridCells*sizeof(uint));

    m_hCellEnd = new uint[m_numGridCells];
    memset(m_hCellEnd, 0, m_numGridCells*sizeof(uint));

    // allocate CPU data

    particle_Hash_Id = new std::vector<std::pair<int, int> >(m_numParticles);
    particle_Hash_Id->clear();

    m_hSortedPos = new float[m_numParticles*4];
    m_hSortedVel = new float[m_numParticles*4];

    memset(m_hSortedPos, 0,m_numParticles*4*sizeof(float));
    memset(m_hSortedVel, 0,m_numParticles*4*sizeof(float));

    // Device memory allocation
	checkCudaErrors(cudaMalloc((void **)&m_dScalarField, m_numGridCells*sizeof(float)));

    sdkCreateTimer(&m_timer);

    m_bInitialized = true;
}


void ParticleSystemCpu::_finalize()
{
    assert(m_bInitialized);

    delete [] m_hPos;
    delete [] m_hVel;
    delete [] m_hCellStart;
    delete [] m_hCellEnd;
    delete [] m_hSortedPos;
    delete [] m_hSortedVel;

	//freeArray(m_dScalarField);
	checkCudaErrors(cudaFree(m_dScalarField));
	
    checkCudaErrors(cudaFree(m_cudaPosVBO));
    checkCudaErrors(cudaFree(m_cudaColorVBO));

}


void sortParticlesCpu(uint *gridParticleHash, uint *gridParticleIndex, uint numParticles)
{
	//std::vector<std::pair<K,V>> items;
	//std::vector<std::pair<gridParticleIndex, gridParticleHash>> items;
	//std::sort(items.begin(), items.end());
}


void  ParticleSystemCpu::integrateSystemCpu(float deltaTime)
{

    for(int n=0; n<m_numParticles; n++)
    {
        float3 pos = getFloat3(m_hPos,n);
        float3 vel = getFloat3(m_hVel,n);

        vel += m_params.gravity * deltaTime;
        vel *= m_params.globalDamping;

        // new position = old position + velocity * deltaTime
		pos += (vel * deltaTime);

        // set this to zero to disable collisions with cube sides
#if 1

        if (pos.x > 1.0f - m_params.particleRadius)
        {
            pos.x = 1.0f - m_params.particleRadius;
            vel.x *= m_params.boundaryDamping;
        }

        if (pos.x < -1.0f + m_params.particleRadius)
        {
            pos.x = -1.0f + m_params.particleRadius;
            vel.x *= m_params.boundaryDamping;
        }



        if (pos.y > 1.0f - m_params.particleRadius)
        {
            pos.y = 1.0f - m_params.particleRadius;
            vel.y *= m_params.boundaryDamping;
        }

        if (pos.z > 1.0f - m_params.particleRadius)
        {
            pos.z = 1.0f - m_params.particleRadius;
            vel.z *= m_params.boundaryDamping;
        }

        if (pos.z < -1.0f + m_params.particleRadius)
        {
            pos.z = -1.0f + m_params.particleRadius;
            vel.z *= m_params.boundaryDamping;
        }

#endif

        if (pos.y < -1.0f + m_params.particleRadius)
        {
            pos.y = -1.0f + m_params.particleRadius;
            vel.y *= m_params.boundaryDamping;
        }

        // store new position and velocity
        setFloat(m_hPos, n, pos);
        setFloat(m_hVel, n, vel);

    }
}


void ParticleSystemCpu::selfConsistencyCheck()
{
    //for(int n=0; n<m_numParticles; n++)
    //{
    //	float3 pos = make_float3(m_hPos[4*n], m_hPos[4*n+1], m_hPos[4*n+2]);
    //
    //	if( (pos.x>1.5f) || (pos.x<-1.5f) )
	//	{
    //		printf("Self consistency check fail on particle %d at (%.1f,%.1f,%.1f)! \n",
    //				n, pos.x, pos.y, pos.z);
    //		break;
	//	}
    //	if( (pos.y>1.5f) || (pos.y<-1.5f) )
	//	{
    //		printf("Self consistency check fail on particle %d at (%.1f,%.1f,%.1f)! \n",
    //				n, pos.x, pos.y, pos.z);
    //		break;
	//	}
    //	if( (pos.z>1.5f) || (pos.z<-1.5f) )
	//	{
    //		printf("Self consistency check fail on particle %d at (%.1f,%.1f,%.1f)! \n",
    //				n, pos.x, pos.y, pos.z);
    //		break;
	//	}
    //
	//	if( isnan(pos.x) )
	//	{
	//		printf("Self consistency check fail on particle %d at (%.1f,%.1f,%.1f)! \n",
	//			n, pos.x, pos.y, pos.z);
	//		break;
	//	}
	//	if(  isnan(pos.y)  )
	//	{
	//		printf("Self consistency check fail on particle %d at (%.1f,%.1f,%.1f)! \n",
	//			n, pos.x, pos.y, pos.z);
	//		break;
	//	}
	//	if( isnan(pos.z)  )
	//	{
	//		printf("Self consistency check fail on particle %d at (%.1f,%.1f,%.1f)! \n",
	//		  n, pos.x, pos.y, pos.z);
	//		break;
	//	}
    //}

    float *v;

    v=m_hPos;
    for(int n=0; n < m_numParticles; n++)
    {
    	float3 q = make_float3(v[4*n], v[4*n+1], v[4*n+2]);
		if( isnan(q.x) || isnan(q.y) || isnan(q.z)  )
		{
			printf("Self consistency check fail on particle %d m_hPos (%.1f,%.1f,%.1f)! \n",
				n, q.x, q.y, q.z);
			break;
		}
    }

    v=m_hSortedPos;
    for(int n=0; n < m_numParticles; n++)
    {
    	float3 q = make_float3(v[4*n], v[4*n+1], v[4*n+2]);
		if( isnan(q.x) || isnan(q.y) || isnan(q.z)  )
		{
			printf("Self consistency check fail on particle %d m_hSortedPos (%.1f,%.1f,%.1f)! \n",
				n, q.x, q.y, q.z);
			break;
		}
    }

    v=m_hVel;
    for(int n=0; n < m_numParticles; n++)
    {
    	float3 q = make_float3(v[4*n], v[4*n+1], v[4*n+2]);
		if( isnan(q.x) || isnan(q.y) || isnan(q.z)  )
		{
			printf("Self consistency check fail on particle %d m_hVel (%.1f,%.1f,%.1f)! \n",
				n, q.x, q.y, q.z);
			break;
		}
    }

    v=m_hSortedVel;
    for(int n=0; n < m_numParticles; n++)
    {
    	float3 q = make_float3(v[4*n], v[4*n+1], v[4*n+2]);
		if( isnan(q.x) || isnan(q.y) || isnan(q.z)  )
		{
			printf("Self consistency check fail on particle %d m_hSortedVel (%.1f,%.1f,%.1f)! \n",
				n, q.x, q.y, q.z);
			break;
		}
    }


}


int3 ParticleSystemCpu::calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x - m_params.worldOrigin.x) / m_params.cellSize.x);
    gridPos.y = floor((p.y - m_params.worldOrigin.y) / m_params.cellSize.y);
    gridPos.z = floor((p.z - m_params.worldOrigin.z) / m_params.cellSize.z);
    return gridPos;
}


int3 ParticleSystemCpu::calcGridPos(uint cellId)
{
	int3 gridPos;
	int xy_residual;

	gridPos.z = cellId / (m_params.gridSize.x * m_params.gridSize.y);
	xy_residual = cellId % ( m_params.gridSize.x * m_params.gridSize.y );
	gridPos.y = xy_residual / m_params.gridSize.y;
	gridPos.x = xy_residual % m_params.gridSize.y;

	return gridPos;
}


uint ParticleSystemCpu::calcGridHash(int3 p )
{
    p.x = p.x & (m_params.gridSize.x-1);  // wrap grid, assumes size is power of 2
    p.y = p.y & (m_params.gridSize.y-1);
    p.z = p.z & (m_params.gridSize.z-1);

    uint res = (p.z* m_params.gridSize.y)* m_params.gridSize.x;
    res += (p.y* m_params.gridSize.x);
    res += p.x;

    return res;
}


void ParticleSystemCpu::calcHashCpu()
{
	particle_Hash_Id->clear();

    for(uint index=0; index < m_numParticles; index++)
    {
      float4 p = getFloat4(m_hPos, index);

      int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));

      uint hash = calcGridHash(gridPos);

      particle_Hash_Id->push_back(std::make_pair(hash,index));
    }
}


void ParticleSystemCpu::reorderDataAndFindCellStartCpu()
{
	uint hash, hash_prev;

	// Note particle_Hash_Id has been std::sort'ed
	// So the first value is the cell the second is the particle index


	// Initialise all cells to empty
    memset(m_hCellStart, 0xffffffff, m_numGridCells*sizeof(uint));

    for(uint index=1; index < m_numParticles; index++)
    {
        hash = (particle_Hash_Id->at(index)).first;  // Get the cell hash
        hash_prev = hash;

        if(index > 0)
        {
        	hash_prev = (particle_Hash_Id->at(index-1)).first;
        }

        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell

        if (index == 0 || hash != hash_prev)
        {
        	m_hCellStart[hash] = index;

            if (index > 0)
                m_hCellEnd[hash_prev] = index;
        }

        if (index == m_numParticles - 1)
        {
        	m_hCellEnd[hash] = index + 1;
        }

        // Now use the sorted index to reorder the pos and vel data
        uint sortedIndex = (particle_Hash_Id->at(index)).second;

        m_hSortedPos[4*index]   = m_hPos[4*sortedIndex];
        m_hSortedPos[4*index+1] = m_hPos[4*sortedIndex+1];
        m_hSortedPos[4*index+2] = m_hPos[4*sortedIndex+2];
        m_hSortedPos[4*index+3] = m_hPos[4*sortedIndex+3];

        m_hSortedVel[4*index]   = m_hVel[4*sortedIndex];
        m_hSortedVel[4*index+1] = m_hVel[4*sortedIndex+1];
        m_hSortedVel[4*index+2] = m_hVel[4*sortedIndex+2];
        m_hSortedVel[4*index+3] = m_hVel[4*sortedIndex+3];
    }
}


float3 ParticleSystemCpu::collideSpheres(float3 posA,
		              float3 posB,
                      float3 velA,
                      float3 velB,
                      float radiusA,
                      float radiusB,
                      float attraction)
{
    // calculate relative position
    float3 relPos = posB - posA;

    float dist = length(relPos);
    float collideDist = radiusA + radiusB;

    float3 force = make_float3(0.0f);

    if (dist < collideDist)
    {
        float3 norm = relPos / dist;
        // relative velocity
        float3 relVel = velB - velA;
        // relative tangential velocity
        float3 tanVel = relVel - (dot(relVel, norm) * norm);
        // spring force
        force = -m_params.spring*(collideDist - dist) * norm;
        // dashpot (damping) force
        force += m_params.damping*relVel;
        // tangential shear force
        force += m_params.shear*tanVel;
        // attraction
        force += attraction*relPos;
    }

    return force;
}


float3 ParticleSystemCpu::collideCell(int3 gridPos,
		           uint index,
                   float3  pos,
                   float3  vel)
{


    uint gridHash = calcGridHash(gridPos);

    if(gridHash >= m_numGridCells)
    	return make_float3(0.0f);

    // get start of bucket for this cell
    uint startIndex = m_hCellStart[gridHash];

    float3 force = make_float3(0.0f);

    if (startIndex != 0xffffffff)          // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = m_hCellEnd[gridHash];

        for (uint j=startIndex; j<endIndex; j++)
        {
            if (j != index)                // check not colliding with self
            {
                float3 pos2 = getFloat3(m_hSortedPos,j);
                float3 vel2 = getFloat3(m_hSortedVel,j);

                // collide two spheres
                force += collideSpheres(pos,
                		pos2,
                		vel,
                		vel2,
                		m_params.particleRadius,
                		m_params.particleRadius,
                		m_params.attraction);
            }
        }
    }

    return force;
}


void ParticleSystemCpu::collideCpu()
{
    for(uint index = 0; index< m_numParticles; index++)
    {

      float3 pos = getFloat3(m_hSortedPos,index);
      float3 vel = getFloat3(m_hSortedVel,index);

      // get address in grid
      int3 gridPos = calcGridPos(pos);

      // examine neighbouring cells
      float3 force = make_float3(0.0f);

      for (int z=-1; z<=1; z++)
      {
          for (int y=-1; y<=1; y++)
          {
              for (int x=-1; x<=1; x++)
              {
                  int3 neighbourPos = gridPos + make_int3(x, y, z);
                  force += collideCell(neighbourPos, index, pos, vel);
              }
          }
      }

      // write new velocity back to original unsorted location

      uint originalIndex = particle_Hash_Id->at(index).second;

      //newVel[originalIndex] = make_float4(vel + force, 0.0f);
      //m_hVel[4*index]   = vel.x + force.x;
      //m_hVel[4*index+1] = vel.y + force.y;
      //m_hVel[4*index+2] = vel.z + force.z;


      setFloat(m_hVel, originalIndex, vel + force);
    }
}


float scalarfieldCpu( float3 meausrePos, float3 particlePos)
{	float3 del = meausrePos - particlePos;
	float r2 = (del.x*del.x) + (del.y*del.y) + (del.z*del.z);
	return 1.0f/(1.0f+r2);
}


float ParticleSystemCpu::get_field(	float3 measurePosition,  // input: absolute measure point coordinates
					int3   neighbourPos)  // input: near by cell [x,y,z] position
{

	uint gridHash = calcGridHash(neighbourPos);
	uint startIndex = m_hCellStart[gridHash];

	float4 *oldPos = (float4*)m_hSortedPos;

	float field = 0;

	if (startIndex != 0xffffffff)          // cell is not empty
	{
		uint endIndex = m_hCellEnd[gridHash];
		for (uint j = startIndex; j<endIndex; j++)
		{
			float3 particlePos = make_float3(oldPos[j]);
			field += scalarfieldCpu(measurePosition, particlePos);
		}
	}
	return field;
}


void ParticleSystemCpu::particleFieldCpu()
{

	for(uint cell_idx=0; cell_idx<m_numGridCells; cell_idx++)
	{

		// get the grid (x,y,z) position of the measure point
		int3 gridPos = calcGridPos(cell_idx);

		// Make the measurement point the centre of this cell
		float3 measurePos = make_float3(0.0f);
		measurePos = make_float3(gridPos);
		measurePos.x = (measurePos.x ) * m_params.cellSize.x;
		measurePos.y = (measurePos.y ) * m_params.cellSize.y;
		measurePos.z = (measurePos.z ) * m_params.cellSize.z;
		measurePos = measurePos + m_params.worldOrigin;

		// examine near-by cells that will contribute to the field at measure point
		float local_field = 0;

		/*
		 * We are trying to create field that is large near particles and dies away very
		 * quickly to zero the further you are from them.
		 *
		 * So we only examine particles that are at most as far away as one adjacent cell.
		 * So for every cell we only consider the nearest neighbour cells.
		 */

		for (int z = -1; z <= 1; z++)
		{
			for (int y = -1; y <= 1; y++)
			{
				for (int x = -1; x <= 1; x++)
				{
					int3 neighbourPos = gridPos + make_int3(x, y, z);

					// Boundary value checking
					if (neighbourPos.x < 0) continue;
					if (neighbourPos.y < 0) continue;
					if (neighbourPos.z < 0) continue;
					if (neighbourPos.x >= m_params.gridSize.x) continue;
					if (neighbourPos.y >= m_params.gridSize.y) continue;
					if (neighbourPos.z >= m_params.gridSize.z) continue;

					local_field += get_field(measurePos, neighbourPos);
				}
			}
		}

		m_hScalarField[cell_idx] = local_field;
	}
}


void ParticleSystemCpu::clampFieldCpu()
{

	for(uint idx=0; idx<m_numGridCells; idx++)
	{
		int3 pos = calcGridPos(idx);

		uint x_face_low(0), y_face_low(0), z_face_low(0);
		uint x_face_high(0), y_face_high(0), z_face_high(0);

		if(pos.x == 0) x_face_low = 1;
		if(pos.y == 0) y_face_low = 1;
		if(pos.z == 0) z_face_low = 1;

		if(pos.x == m_params.gridSize.x ) x_face_high = 1;
		if(pos.y == m_params.gridSize.y ) y_face_high = 1;
		if(pos.z == m_params.gridSize.z ) z_face_high = 1;

		uint face = x_face_low | y_face_low | z_face_low;
		face |= (x_face_high | y_face_high | z_face_high);


		if(!face)
			return;

		if(m_hScalarField[idx] > 10.0)
			m_hScalarField[idx]  = 9.99;
	}
}


void ParticleSystemCpu::update(float deltaTime, bool updateField)
{
    assert(m_bInitialized);

    integrateSystemCpu(deltaTime);

    calcHashCpu();

    std::sort(particle_Hash_Id->begin(), particle_Hash_Id->end());

    reorderDataAndFindCellStartCpu();

    collideCpu();

    if(updateField)
    {

	  particleFieldCpu();

	  //clampFieldCpu();

      /*
       * Transfer field data to the GPU for marching cubes ...
       */

	  checkCudaErrors(cudaMemcpy((char *) m_dScalarField ,
			  m_hScalarField, m_numGridCells*sizeof(float), cudaMemcpyHostToDevice));
    }
}


void ParticleSystemCpu::dumpGrid()
{
    // dump grid information
    uint maxCellSize = 0;

    for (uint i=0; i<m_numGridCells; i++)
    {
        if (m_hCellStart[i] != 0xffffffff)
        {
            uint cellSize = m_hCellEnd[i] - m_hCellStart[i];

            //            printf("cell: %d, %d particles\n", i, cellSize);
            if (cellSize > maxCellSize)
            {
                maxCellSize = cellSize;
            }
        }
    }

    printf("maximum particles per cell = %d\n", maxCellSize);
}


float* ParticleSystemCpu::dumpField()
{
	//copyArrayFromDevice(m_hScalarField, m_dScalarField, 0, sizeof(float)*m_numGridCells);
	return m_hScalarField;
}


float *ParticleSystemCpu::getField()
{
  fprintf(stderr, "Giving back the scalar field!\n");
  return m_dScalarField;
}

void ParticleSystemCpu::dumpParticles(uint start, uint count)
{
    for (uint n=start; n<start+count; n++)
    {
        //        printf("%d: ", i);
    	float3 pos = make_float3(m_hPos[4*n], m_hPos[4*n+1], m_hPos[4*n+2]);
    	printf("pos: (%.4f, %.4f, %.4f)\n", pos.x, pos.y, pos.z);
        //printf("pos: (%.4f, %.4f, %.4f, %.4f)\n", m_hPos[i*4+0], m_hPos[i*4+1], m_hPos[i*4+2], m_hPos[i*4+3]);
        //printf("vel: (%.4f, %.4f, %.4f, %.4f)\n", m_hVel[i*4+0], m_hVel[i*4+1], m_hVel[i*4+2], m_hVel[i*4+3]);
    }
}


float *ParticleSystemCpu::getArray(ParticleArray array)
{
    assert(m_bInitialized);

	printf("Calling getArray() \n");

    float *hdata = 0;

    switch (array)
    {
        default:
        case POSITION:
            return m_hPos;

        case VELOCITY:
            return m_hVel;
    }

    return hdata;
}


void ParticleSystemCpu::setArray(ParticleArray array, const float *data, int start, int count)
{
    assert(m_bInitialized);
}


inline float frand()
{
    return rand() / (float) RAND_MAX;
}


void ParticleSystemCpu::initGrid(uint *size, float spacing, float jitter, uint numParticles)
{
    srand(1973);

    for (uint z=0; z<size[2]; z++)
    {
        for (uint y=0; y<size[1]; y++)
        {
            for (uint x=0; x<size[0]; x++)
            {
                uint i = (z*size[1]*size[0]) + (y*size[0]) + x;

                if (i < numParticles)
                {
                    m_hPos[4*i]   = (spacing * x) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[4*i+1] = (spacing * y) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[4*i+2] = (spacing * z) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[4*i+3] = 1.0f;

                    m_hVel[4*i]   = 0.0f;
                    m_hVel[4*i+1] = 0.0f;
                    m_hVel[4*i+2] = 0.0f;
                    m_hVel[4*i+3] = 0.0f;
                }
            }
        }
    }
}


void ParticleSystemCpu::reset(ParticleConfig config)
{
    switch (config)
    {
        default:
        case CONFIG_RANDOM:
            {
                int p = 0, v = 0;

                for (uint i=0; i < m_numParticles; i++)
                {
                    float point[3];
                    point[0] = frand();
                    point[1] = frand();
                    point[2] = frand();
                    m_hPos[p++] = 2 * (point[0] - 0.5f);
                    m_hPos[p++] = 2 * (point[1] - 0.5f);
                    m_hPos[p++] = 2 * (point[2] - 0.5f);
                    m_hPos[p++] = 1.0f; // radius
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                }
            }
            break;

        case CONFIG_GRID:
            {
                float jitter = m_params.particleRadius*0.01f;
                uint s = (int) ceilf(powf((float) m_numParticles, 1.0f / 3.0f));
                uint gridSize[3];
                gridSize[0] = gridSize[1] = gridSize[2] = s;
                initGrid(gridSize, m_params.particleRadius*2.0f, jitter, m_numParticles);
            }
            break;
    }

    setArray(POSITION, m_hPos, 0, m_numParticles);
    setArray(VELOCITY, m_hVel, 0, m_numParticles);

}


void ParticleSystemCpu::addSphere(int start, float *pos, float *vel, int r, float spacing)
{
    uint index = start;

    for (int z=-r; z<=r; z++)
    {
        for (int y=-r; y<=r; y++)
        {
            for (int x=-r; x<=r; x++)
            {
                float dx = x*spacing;
                float dy = y*spacing;
                float dz = z*spacing;
                float l = sqrtf(dx*dx + dy*dy + dz*dz);
                float jitter = m_params.particleRadius*0.01f;

                if ((l <= m_params.particleRadius*2.0f*r) && (index < m_numParticles))
                {
                    m_hPos[index*4]   = pos[0] + dx + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[index*4+1] = pos[1] + dy + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[index*4+2] = pos[2] + dz + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[index*4+3] = pos[3];

                    m_hVel[index*4]   = vel[0];
                    m_hVel[index*4+1] = vel[1];
                    m_hVel[index*4+2] = vel[2];
                    m_hVel[index*4+3] = vel[3];
                    index++;
                }
            }
        }
    }

    setArray(POSITION, m_hPos, start, index);
    setArray(VELOCITY, m_hVel, start, index);
}
