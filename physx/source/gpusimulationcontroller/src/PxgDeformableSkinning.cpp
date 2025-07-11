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

#include "PxgDeformableSkinning.h"

#include "foundation/PxUserAllocated.h"

#include "PxPhysXGpu.h"
#include "PxgKernelWrangler.h"
#include "PxgKernelIndices.h"
#include "GuAABBTree.h"
#include "foundation/PxMathUtils.h"


namespace physx
{
	PxgDeformableSkinning::PxgDeformableSkinning(PxgKernelLauncher& kernelLauncher)
	{
		mKernelLauncher = kernelLauncher;
	}

	void PxgDeformableSkinning::computeNormalVectors(
		PxTrimeshSkinningGpuData* skinningDataArrayD, PxU32 arrayLength,
		hipStream_t stream, PxU32 numGpuThreads)
	{
		physx::PxScopedCudaLock _lock(*mKernelLauncher.getCudaContextManager());

		const PxU32 numThreadsPerBlock = 256;
		const PxU32 numBlocks = (numGpuThreads + numThreadsPerBlock - 1) / numThreadsPerBlock;

		const PxU32 numThreadsPerBlockSmallKernels = 1024;
		const PxU32 numBlocksSmallKernels = (numGpuThreads + numThreadsPerBlockSmallKernels - 1) / numThreadsPerBlockSmallKernels;

		mKernelLauncher.launchKernelXYZ(PxgKernelIds::util_ZeroNormals, numBlocksSmallKernels, arrayLength, 1, numThreadsPerBlockSmallKernels, 1, 1, 0, stream,
			skinningDataArrayD);

		mKernelLauncher.launchKernelXYZ(PxgKernelIds::util_ComputeNormals, numBlocks, arrayLength, 1, numThreadsPerBlock, 1, 1, 0, stream,
			skinningDataArrayD);

		mKernelLauncher.launchKernelXYZ(PxgKernelIds::util_NormalizeNormals, numBlocksSmallKernels, arrayLength, 1, numThreadsPerBlockSmallKernels, 1, 1, 0, stream,
			skinningDataArrayD);
	}

	void PxgDeformableSkinning::evaluateVerticesEmbeddedIntoSurface(
		PxTrimeshSkinningGpuData* skinningDataArrayD, PxU32 arrayLength,
		hipStream_t stream, PxU32 numGpuThreads)
	{
		physx::PxScopedCudaLock _lock(*mKernelLauncher.getCudaContextManager());
		const PxU32 numThreadsPerBlock = 256;
		const PxU32 numBlocks = (numGpuThreads + numThreadsPerBlock - 1) / numThreadsPerBlock;
		mKernelLauncher.launchKernelXYZ(PxgKernelIds::util_InterpolateSkinnedClothVertices, numBlocks, arrayLength, 1, numThreadsPerBlock, 1, 1, 0, stream,
			skinningDataArrayD);
	}

	void PxgDeformableSkinning::evaluateVerticesEmbeddedIntoVolume(
		PxTetmeshSkinningGpuData* skinningDataArrayD, PxU32 arrayLength,
		hipStream_t stream, PxU32 numGpuThreads)
	{
		physx::PxScopedCudaLock _lock(*mKernelLauncher.getCudaContextManager());
		const PxU32 numThreadsPerBlock = 256;
		const PxU32 numBlocks = (numGpuThreads + numThreadsPerBlock - 1) / numThreadsPerBlock;
		mKernelLauncher.launchKernelXYZ(PxgKernelIds::util_InterpolateSkinnedSoftBodyVertices, numBlocks, arrayLength, 1, numThreadsPerBlock, 1, 1, 0, stream,
			skinningDataArrayD);
	}
}
