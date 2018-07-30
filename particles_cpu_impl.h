

#ifndef PARTICLES_CPU_H
#define PARTICLES_CPU_H


#include <math.h>
#include "helper_math.h"
#include "math_constants.h"



void calcHashH(uint *gridParticleHash,  // output
               uint   *gridParticleIndex, // output
               float4 *pos,               // input: positions
               uint    numParticles);


void reorderDataAndFindCellStartH(uint   *cellStart,        // output: cell start index
                                  uint   *cellEnd,          // output: cell end index
                                  float4 *sortedPos,        // output: sorted positions
                                  float4 *sortedVel,        // output: sorted velocities
                                  uint   *gridParticleHash, // input:  sorted grid hashes
                                  uint   *gridParticleIndex,// input:  sorted particle indices
                                  float4 *oldPos,           // input:  sorted position array
                                  float4 *oldVel,           // input:  sorted velocity array
                                  uint    numParticles);

void collideH(float4 *newVel,             // output: new velocity
              float4 *oldPos,             // input:  sorted positions
              float4 *oldVel,             // input:  sorted velocities
              uint   *gridParticleIndex,  // input:  sorted particle indices
              uint   *cellStart,
              uint   *cellEnd,
              uint    numParticles);


#endif /* PARTICLES_CPU_H */
