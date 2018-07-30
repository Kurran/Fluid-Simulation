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

#ifndef __PARTICLESYSTEM_GPU_H__
#define __PARTICLESYSTEM_GPU_H__

#define DEBUG_GRID 0
#define DO_TIMING 0

#include <helper_functions.h>
#include "particleSystem.h"
#include "particles_kernel.cuh"
#include "vector_functions.h"
#include "defines.h"

// Particle system class
class ParticleSystemGpu : public ParticleSystem
{
    public:
        ParticleSystemGpu(uint numParticles, uint3 gridSize, bool bUseOpenGL);
        ~ParticleSystemGpu();


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
        bool m_bInitialized;


        uint   m_gridSortBits;

        float *m_dVectorField;
        float *m_hVectorField;


        struct cudaGraphicsResource *m_cuda_posvbo_resource; // handles OpenGL-CUDA exchange
        struct cudaGraphicsResource *m_cuda_colorvbo_resource; // handles OpenGL-CUDA exchange

        uint  m_numGridCells;

};

#endif // __PARTICLESYSTEM_GPU_H__
