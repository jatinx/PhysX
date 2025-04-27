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

#ifndef PX_CUDA_CONTEX_H
#define PX_CUDA_CONTEX_H

#include "foundation/PxPreprocessor.h"

#if PX_SUPPORT_GPU_PHYSX

#include "PxCudaTypes.h"

#if !PX_DOXYGEN
namespace physx
{
#endif
	struct PxCudaKernelParam
	{
		void* data;
		size_t size;
	};

	// workaround for not being able to forward declare enums in PxCudaTypes.h. 
	// provides different automatic casting depending on whether hip/hip_runtime.h was included beforehand or not.
	template<typename CUenum>
	struct PxCUenum
	{
		PxU32 value;

		PxCUenum(CUenum e) { value = PxU32(e); }
		operator CUenum() const { return CUenum(value); }
	};

#ifdef CUDA_VERSION
	typedef PxCUenum<hipJitOption> PxCUjit_option;
	typedef PxCUenum<hipError_t> PxCUresult;
#else
	typedef PxCUenum<PxU32> PxCUjit_option;
	typedef PxCUenum<PxU32> PxCUresult;
#endif

#define PX_CUDA_KERNEL_PARAM(X)		{ (void*)&X, sizeof(X) }
#define PX_CUDA_KERNEL_PARAM2(X)	(void*)&X

	class PxDeviceAllocatorCallback;
	/**
	Cuda Context
	*/
	class PxCudaContext
	{
	protected:
		virtual ~PxCudaContext() {}

		PxDeviceAllocatorCallback* mAllocatorCallback;

	public:
		virtual void release() = 0;

		virtual PxCUresult memAlloc(hipDeviceptr_t *dptr, size_t bytesize) = 0;

		virtual PxCUresult memFree(hipDeviceptr_t dptr) = 0;

		virtual PxCUresult memHostAlloc(void **pp, size_t bytesize, unsigned int Flags) = 0;

		virtual PxCUresult memFreeHost(void *p) = 0;

		virtual PxCUresult memHostGetDevicePointer(hipDeviceptr_t *pdptr, void *p, unsigned int Flags) = 0;

		virtual PxCUresult moduleLoadDataEx(hipModule_t *module, const void *image, unsigned int numOptions, PxCUjit_option *options, void **optionValues) = 0;

		virtual PxCUresult moduleGetFunction(hipFunction_t *hfunc, hipModule_t hmod, const char *name) = 0;

		virtual PxCUresult moduleUnload(hipModule_t hmod) = 0;

		virtual PxCUresult streamCreate(hipStream_t *phStream, unsigned int Flags) = 0;

		virtual PxCUresult streamCreateWithPriority(hipStream_t *phStream, unsigned int flags, int priority) = 0;

		virtual PxCUresult streamFlush(hipStream_t hStream) = 0;

		virtual PxCUresult streamWaitEvent(hipStream_t hStream, hipEvent_t hEvent, unsigned int Flags) = 0;

		virtual PxCUresult streamWaitEvent(hipStream_t hStream, hipEvent_t hEvent) = 0;

		virtual PxCUresult streamDestroy(hipStream_t hStream) = 0;

		virtual PxCUresult streamSynchronize(hipStream_t hStream) = 0;

		virtual PxCUresult eventCreate(hipEvent_t *phEvent, unsigned int Flags) = 0;

		virtual PxCUresult eventRecord(hipEvent_t hEvent, hipStream_t hStream) = 0;

		virtual PxCUresult eventQuery(hipEvent_t hEvent) = 0;

		virtual PxCUresult eventSynchronize(hipEvent_t hEvent) = 0;

		virtual PxCUresult eventDestroy(hipEvent_t hEvent) = 0;

		virtual PxCUresult launchKernel(
			hipFunction_t f,
			unsigned int gridDimX,
			unsigned int gridDimY,
			unsigned int gridDimZ,
			unsigned int blockDimX,
			unsigned int blockDimY,
			unsigned int blockDimZ,
			unsigned int sharedMemBytes,
			hipStream_t hStream,
			PxCudaKernelParam* kernelParams,
			size_t kernelParamsSizeInBytes,
			void** extra,
			const char* file,
			int line
		) = 0;

		// PT: same as above but without copying the kernel params to a local stack before the launch
		// i.e. the kernelParams data is passed directly to the kernel.
		virtual PxCUresult launchKernel(
			hipFunction_t f,
			PxU32 gridDimX, PxU32 gridDimY, PxU32 gridDimZ,
			PxU32 blockDimX, PxU32 blockDimY, PxU32 blockDimZ,
			PxU32 sharedMemBytes,
			hipStream_t hStream,
			void** kernelParams,
			void** extra,
			const char* file,
			int line
		) = 0;

		virtual PxCUresult memcpyDtoH(void *dstHost, hipDeviceptr_t srcDevice, size_t ByteCount) = 0;

		virtual PxCUresult memcpyDtoHAsync(void* dstHost, hipDeviceptr_t srcDevice, size_t ByteCount, hipStream_t hStream) = 0;

		virtual PxCUresult memcpyHtoD(hipDeviceptr_t dstDevice, const void *srcHost, size_t ByteCount) = 0;

		virtual PxCUresult memcpyHtoDAsync(hipDeviceptr_t dstDevice, const void *srcHost, size_t ByteCount, hipStream_t hStream) = 0;

		virtual PxCUresult memcpyDtoDAsync(hipDeviceptr_t dstDevice, hipDeviceptr_t srcDevice, size_t ByteCount, hipStream_t hStream) = 0;

		virtual PxCUresult memcpyDtoD(hipDeviceptr_t dstDevice, hipDeviceptr_t srcDevice, size_t ByteCount) = 0;

		virtual PxCUresult memcpyPeerAsync(hipDeviceptr_t dstDevice, hipCtx_t dstContext, hipDeviceptr_t srcDevice, hipCtx_t srcContext, size_t ByteCount, hipStream_t hStream) = 0;

		virtual PxCUresult memsetD32Async(hipDeviceptr_t dstDevice, unsigned int ui, size_t N, hipStream_t hStream) = 0;

		virtual PxCUresult memsetD8Async(hipDeviceptr_t dstDevice, unsigned char uc, size_t N, hipStream_t hStream) = 0;

		virtual PxCUresult memsetD32(hipDeviceptr_t dstDevice, unsigned int ui, size_t N) = 0;

		virtual PxCUresult memsetD16(hipDeviceptr_t dstDevice, unsigned short uh, size_t N) = 0;

		virtual PxCUresult memsetD8(hipDeviceptr_t dstDevice, unsigned char uc, size_t N) = 0;

		virtual PxCUresult getLastError() = 0;

		PxDeviceAllocatorCallback* getAllocatorCallback() { return mAllocatorCallback; }

		virtual void setAbortMode(bool abort) = 0;

		virtual bool isInAbortMode() = 0;
	};

#if !PX_DOXYGEN
} // namespace physx
#endif

#endif // PX_SUPPORT_GPU_PHYSX
#endif

