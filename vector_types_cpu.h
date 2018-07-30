/*
 * vector_types_cpu.h
 *
 *  Created on: 1 Mar 2016
 *      Author: harpal
 */

#ifndef VECTOR_TYPES_CPU_H_
#define VECTOR_TYPES_CPU_H_


struct   int3
{
    int x, y, z;
};

struct   uint3
{
    unsigned int x, y, z;
};

struct int4
{
    int x, y, z, w;
};

struct uint4
{
    unsigned int x, y, z, w;
};

struct   float3
{
    float x, y, z;
};

struct float4
{
    float x, y, z, w;
};


#endif /* VECTOR_TYPES_CPU_H_ */
