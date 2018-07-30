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

#ifndef __PARTICLESYSTEM_CPU_H__
#define __PARTICLESYSTEM_CPU_H__

#define DEBUG_GRID 0
#define DO_TIMING 0

#include <vector>
#include <utility>
#include <algorithm>

#include <helper_functions.h>

#include "particleSystem.h"
#include "helper_math.h"
#include "defines.h"

// Particle system class
class ParticleSystemCpu : public ParticleSystem
{
    public:
        ParticleSystemCpu(uint numParticles, uint3 gridSize, bool bUseOpenGL);
        ~ParticleSystemCpu();
        void selfConsistencyCheck();

        void update(float deltaTime, bool updateField);
        void reset(ParticleConfig config);
        float *getArray(ParticleArray array);
        void setArray(ParticleArray array, const float *data, int start, int count);
        void dumpGrid();
        void dumpParticles(uint start, uint count);
        float* dumpField();
        void addSphere(int index, float *pos, float *vel, int r, float spacing);
        float* getField();

    protected: // methods
        //ParticleSystemGpu() {};
        uint createVBO(uint size);

        void _initialize(int numParticles);
        void _finalize();

        void initGrid(uint *size, float spacing, float jitter, uint numParticles);



    protected: // data

        uint calcGridHash(int3 gridPos);

        void calcHashCpu();

        int3 calcGridPos(float3 p);

        int3 calcGridPos(uint cellId);

        void integrateSystemCpu(float deltaTime);

        void reorderDataAndFindCellStartCpu();

        float3 collideCell(int3 gridPos, uint index);

        void collideCpu();

        void particleFieldCpu();


        float get_field( float3 measurePosition,
        				 int3   neighbourPos);

        void clampFieldCpu();

        float3 collideSpheres(float3 posA,
        		              float3 posB,
                              float3 velA,
                              float3 velB,
                              float radiusA,
                              float radiusB,
                              float attraction);

        float3 collideCell(int3 gridPos,
        		           uint index,
                           float3  pos,
                           float3  vel);


        bool m_bInitialized;
        uint m_gridSortBits;

        /*
         * It's efficient to store the particle id  and hash values as a vector of
         * pairs.
         */

        std::vector< std::pair<int, int> > *particle_Hash_Id;

        struct cudaGraphicsResource *m_cuda_posvbo_resource; // handles OpenGL-CUDA exchange
        struct cudaGraphicsResource *m_cuda_colorvbo_resource; // handles OpenGL-CUDA exchange

        uint  m_numGridCells;

};

#endif // __PARTICLESYSTEM_CPU_H__
