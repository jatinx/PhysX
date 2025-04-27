#include "hip/hip_runtime.h"
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.

#ifndef PXG_PBD_PARTICLE_SYSTEM_CORE_H
#define PXG_PBD_PARTICLE_SYSTEM_CORE_H

#include "foundation/PxVec3.h"
#include "foundation/PxSimpleTypes.h"

#include "cudamanager/PxCudaTypes.h"

#include "PxgParticleSystemCore.h"

namespace physx
{
	class PxCudaContextManager;
	class PxgBodySimManager;
	class PxgCudaKernelWranglerManager;
	class PxgGpuContext;
	struct PxGpuParticleBufferIndexPair;
	class PxgHeapMemoryAllocatorManager;
	class PxgParticleAndDiffuseBuffer;
	class PxgParticleClothBuffer;
	class PxgParticleRigidBuffer;
	class PxgParticleSystem;
	class PxgParticleSystemBuffer;
	class PxgParticleSystemDiffuseBuffer;
	class PxgSimulationController;

	namespace Dy
	{
		class ParticleSystemCore;
	}

	class PxgPBDParticleSystemCore : public PxgParticleSystemCore, public PxgDiffuseParticleCore
	{
	public:
		PxgPBDParticleSystemCore(PxgCudaKernelWranglerManager* gpuKernelWrangler, PxCudaContextManager* cudaContextManager,
			PxgHeapMemoryAllocatorManager* heapMemoryManager, PxgSimulationController* simController,
			PxgGpuContext* gpuContext, PxU32 maxParticleContacts);
		virtual ~PxgPBDParticleSystemCore();


		// calculate AABB bound for each particle volumes
		void updateVolumeBound(hipDeviceptr_t particleSystemsd, hipDeviceptr_t activeParticleSystemsd, const PxU32 numActiveParticleSystems,
			hipStream_t bpStream);

		virtual void preIntegrateSystems(const PxU32 nbActiveParticleSystems, const PxVec3 gravity, const PxReal dt);
		//virtual void updateBounds(PxgParticleSystem* particleSystems, PxU32* activeParticleSystems, const PxU32 nbActiveParticleSystems);
		virtual void updateGrid();
		virtual void selfCollision();
		//this is for solving selfCollsion and contacts between particles and primitives based on sorted by particle id

		virtual void constraintPrep(hipDeviceptr_t prePrepDescd, hipDeviceptr_t prepDescd, hipDeviceptr_t solverCoreDescd, hipDeviceptr_t sharedDescd,
			const PxReal dt, hipStream_t solverStream, bool isTGS, PxU32 numSolverBodies);
		virtual void updateParticles(const PxReal dt);
		virtual void solve(hipDeviceptr_t prePrepDescd, hipDeviceptr_t solverCoreDescd,
			hipDeviceptr_t sharedDescd, hipDeviceptr_t artiCoreDescd, const PxReal dt, hipStream_t solverStream);

		virtual void solveTGS(hipDeviceptr_t prePrepDescd, hipDeviceptr_t solverCoreDescd,
			hipDeviceptr_t sharedDescd, hipDeviceptr_t artiCoreDescd, const PxReal dt, const PxReal totalInvDt, hipStream_t solverStream,
			const bool isVelocityIteration, PxI32 iterationIndex, PxI32 numTGSIterations, PxReal coefficient);

		virtual void prepParticleConstraint(hipDeviceptr_t prePrepDescd, hipDeviceptr_t prepDescd, hipDeviceptr_t sharedDescd, bool isTGS, const PxReal dt);


		virtual void integrateSystems(const PxReal dt, const PxReal epsilonSq);
		virtual void onPostSolve();
		virtual void gpuMemDmaUpParticleSystem(PxgBodySimManager& bodySimManager, hipStream_t stream);
		virtual void getMaxIterationCount(PxgBodySimManager& bodySimManager, PxI32& maxPosIters, PxI32& maxVelIters);
		virtual void releaseParticleSystemDataBuffer();

		void solveVelocities(hipDeviceptr_t particleSystemsd, hipDeviceptr_t activeParticleSystemsd, const PxU32 nbActiveParticleSystems, const PxReal dt);

		void solveParticleCollision(const PxReal dt, bool isTGS, PxReal coefficient);
		
		virtual void finalizeVelocities(const PxReal dt, const PxReal scale);

		void solveSprings(hipDeviceptr_t particleSystemsd, hipDeviceptr_t activeParticleSystemsd,
			const PxU32 nbActiveParticleSystems, const PxReal dt, bool isTGS);

		void initializeSprings(hipDeviceptr_t particleSystemsd, hipDeviceptr_t activeParticleSystemsd,
			const PxU32 nbActiveParticleSystems);

		// Direct-GPU API
		PX_DEPRECATED void applyParticleBufferDataDEPRECATED(const PxU32* indices, const PxGpuParticleBufferIndexPair* indexPairs, const PxParticleBufferFlags* flags, PxU32 nbUpdatedBuffers, hipEvent_t waitEvent, hipEvent_t signalEvent);

	private:

		void allocateParticleBuffer(const PxU32 nbTotalParticleSystems, hipStream_t stream);
		void allocateParticleDataBuffer(void** bodySimsLL, hipStream_t stream);
		void updateDirtyData(PxgBodySimManager& bodySimManager, hipStream_t stream);

		void resizeParticleDataBuffer(PxgParticleSystem& particleSystem, PxgParticleSystemBuffer* buffer, const PxU32 maxParticles, const PxU32 maxNeighborhood, hipStream_t stream);
		void resizeDiffuseParticleDiffuseBuffer(PxgParticleSystem& particleSystem, PxgParticleSystemDiffuseBuffer* diffuseBuffer, const PxU32 maxDiffuseParticles, hipStream_t stream);
		bool createUserParticleData(PxgParticleSystem& particleSystem, Dy::ParticleSystemCore& dyParticleSystemCore, PxgParticleSystemBuffer* buffer, PxgParticleSystemDiffuseBuffer* diffuseBuffer,
			hipStream_t stream);

		PX_FORCE_INLINE PxU32 getMaxSpringsPerBuffer() { return mMaxSpringsPerBuffer; }
		PX_FORCE_INLINE PxU32 getMaxSpringPartitionsPerBuffer() { return mMaxSpringPartitionsPerBuffer; }
		PX_FORCE_INLINE PxU32 getMaxSpringsPerPartitionPerBuffer() { return mMaxSpringsPerPartitionPerBuffer; }
		PX_FORCE_INLINE PxU32 getMaxRigidsPerBuffer() { return mMaxRigidsPerBuffer; }

		void calculateHashForDiffuseParticles(hipDeviceptr_t particleSystemsd, hipDeviceptr_t activeParticleSystemsd, const PxU32 numActiveParticleSystems);

		void solveDensities(hipDeviceptr_t particleSystemsd, hipDeviceptr_t activeParticleSystemd, const PxU32 nbActiveParticleSystems, const PxReal dt,
			PxReal coefficient);

		void solveInflatables(hipDeviceptr_t particleSystemsd, hipDeviceptr_t activeParticleSystemd, const PxU32 nbActiveParticleSystems, const PxReal coefficient, const PxReal dt);

		void solveShapes(hipDeviceptr_t particleSystemsd, hipDeviceptr_t activeParticleSystemd, const PxU32 nbActiveParticleSystems, const PxReal dt, const PxReal biasCoefficient);

		void solveAerodynamics(hipDeviceptr_t particleSystemsd, hipDeviceptr_t activeParticleSystemd, const PxU32 nbActiveParticleSystems, const PxReal dt);

		void solveDiffuseParticles(hipDeviceptr_t particleSystemsd, hipDeviceptr_t activeParticleSystemd, const PxU32 nbActiveParticleSystems, const PxReal dt);

		//-------------------------------------------------------------------------------
		// Materials
		void updateMaterials(hipDeviceptr_t particleSystemsd, hipDeviceptr_t activeParticleSystemd, const PxU32 nbActiveParticleSystems, hipStream_t bpStream, const PxReal invTotalDt);

		PxU32							mMaxClothBuffersPerSystem;
		PxU32							mMaxClothsPerBuffer;
		PxU32							mMaxSpringsPerBuffer;
		PxU32							mMaxSpringPartitionsPerBuffer;
		PxU32							mMaxSpringsPerPartitionPerBuffer;
		PxU32							mMaxTrianglesPerBuffer;
		PxU32							mMaxVolumesPerBuffer;
		PxU32							mMaxRigidBuffersPerSystem;
		PxU32							mMaxRigidsPerBuffer;//compute the max rigids(shape matching) for each particle system
		PxU32							mMaxNumPhaseToMaterials; //compute the max number of phase to materials for each particle system
		bool							mComputePotentials;
		PxU32							mNumActiveParticleSystems;
	};
}


#endif
