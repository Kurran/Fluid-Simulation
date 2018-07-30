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

#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#define DEBUG_GRID 0
#define DO_TIMING 0

#include <helper_functions.h>
#include "particles_kernel.cuh"
#include "vector_functions.h"
//#include "defines.h"


// Particle system class
class ParticleSystem
{
    public:
        ParticleSystem(uint numParticles, uint3 gridSize, bool bUseOpenGL) :
        	m_numParticles(numParticles),
        	m_gridSize(gridSize),
        	m_bUseOpenGL(bUseOpenGL),
            m_timer(NULL),
            m_solverIterations(1){}

	    //ParticleSystem(){};

        virtual ~ParticleSystem(){};

        enum ParticleConfig
        {
            CONFIG_RANDOM,
            CONFIG_GRID,
            _NUM_CONFIGS
        };

        enum ParticleArray
        {
            POSITION,
            VELOCITY,
        };

        /*
         * Make the update pure virtual
         * Forcing this to be an interface specification
         */

        virtual void update(float deltaTime, bool updateField) = 0;

        virtual void reset(ParticleConfig config) = 0;

        virtual float *getArray(ParticleArray array) = 0;

        virtual void setArray(ParticleArray array, const float *data, int start, int count) = 0;


        virtual void dumpGrid() = 0;

        virtual void dumpParticles(uint start, uint count) = 0;

        virtual float* dumpField() = 0;

        virtual float* getField() = 0;

        void setIterations(int i)
        {
            m_solverIterations = i;
        }
        void setDamping(float x)
        {
            m_params.globalDamping = x;
        }
        void setGravity(float x)
        {
            m_params.gravity = make_float3(0.0f, x, 0.0f);
        }
        void setCollideSpring(float x)
        {
            m_params.spring = x;
        }
        void setCollideDamping(float x)
        {
            m_params.damping = x;
        }
        void setCollideShear(float x)
        {
            m_params.shear = x;
        }
        void setCollideAttraction(float x)
        {
            m_params.attraction = x;
        }
        void setColliderPos(float3 x)
        {
            m_params.colliderPos = x;
        }

        float  getParticleRadius()
        {
            return m_params.particleRadius;
        }
        float3 getColliderPos()
        {
            return m_params.colliderPos;
        }
        float  getColliderRadius()
        {
            return m_params.colliderRadius;
        }
        uint3  getGridSize()
        {
            return m_params.gridSize;
        }
        float3 getWorldOrigin()
        {
            return m_params.worldOrigin;
        }
        float3 getCellSize()
        {
            return m_params.cellSize;
        }
        int getNumParticles() const
        {
        	return m_numParticles;
        }
        unsigned int getCurrentReadBuffer() const
        {
            return m_posVbo;
        }
        unsigned int getColorBuffer() const
        {
            return m_colorVBO;
        }
        void* getCudaPosVBO() const
        {
            return (void *)m_cudaPosVBO;
        }
        void* getCudaColorVBO() const
        {
            return (void *)m_cudaColorVBO;
        }



        virtual void addSphere(int index, float *pos, float *vel, int r, float spacing) = 0;

    protected:

        bool m_bUseOpenGL;
        uint m_numParticles;


        // CPU data
        float *m_hPos;              // particle positions
        float *m_hVel;              // particle velocities
        float *m_hSortedPos;
        float *m_hSortedVel;
        uint  *m_hCellStart;
        uint  *m_hCellEnd;

        // GPU data
        float *m_dPos;
        float *m_dVel;
        float *m_dSortedPos;
        float *m_dSortedVel;
        uint  *m_dGridParticleHash; // grid hash value for each particle
        uint  *m_dGridParticleIndex;// particle index for each particle
        uint  *m_dCellStart;        // index of start of each cell in sorted list
        uint  *m_dCellEnd;          // index of end of cell


        // params
        SimParams m_params;
        uint3 m_gridSize;
        uint  m_solverIterations;

        StopWatchInterface *m_timer;

        uint   m_posVbo;            // vertex buffer object for particle positions
        uint   m_colorVBO;          // vertex buffer object for colors

        float *m_cudaPosVBO;        // these are the CUDA deviceMem Pos
        float *m_cudaColorVBO;      // these are the CUDA deviceMem Color

		// grid data for isofiledfield method
        float *m_hScalarField;
		float *m_dScalarField;

};

#endif // __PARTICLESYSTEM_H__
