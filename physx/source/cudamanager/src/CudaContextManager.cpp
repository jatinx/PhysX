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

#include "foundation/PxAssert.h"
#include "foundation/PxAtomic.h"
#include "foundation/PxErrorCallback.h"
#include "foundation/PxMath.h"
#include "foundation/PxPreprocessor.h"
#include "foundation/PxMutex.h"
#include "foundation/PxThread.h"
#include "foundation/PxUserAllocated.h"
#include "foundation/PxString.h"
#include "foundation/PxAlloca.h"
#include "foundation/PxArray.h"

#include "PhysXDeviceSettings.h"

// from the point of view of this source file the GPU library is linked statically
#ifndef PX_PHYSX_GPU_STATIC
	#define PX_PHYSX_GPU_STATIC
#endif
#include "PxPhysXGpu.h"

#if PX_LINUX && PX_CLANG
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdocumentation"
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
#endif
#include <hip/hip_runtime.h>
#if PX_LINUX && PX_CLANG
#pragma clang diagnostic pop
#endif

#include "cudamanager/PxCudaContextManager.h"
#include "cudamanager/PxCudaContext.h"

#if PX_WIN32 || PX_WIN64

#include "foundation/windows/PxWindowsInclude.h"


class IDirect3DDevice9;
class IDirect3DResource9;
class IDirect3DVertexBuffer9;
#include <cudad3d9.h>

class IDXGIAdapter;
class ID3D10Device;
class ID3D10Resource;
#include <cudad3d10.h>

struct ID3D11Device;
struct ID3D11Resource;
#include <cudad3d11.h>

#endif // PX_WINDOWS_FAMILY

#if PX_LINUX
#include <dlfcn.h>
static void* GetProcAddress(void* handle, const char* name) { return dlsym(handle, name); }
#endif

// Defining these instead of including gl.h eliminates a dependency
typedef unsigned int GLenum;
typedef unsigned int GLuint;

//#include <GL/gl.h>
#include <cudaGL.h>
#include <assert.h>

#include "foundation/PxErrors.h"
#include "foundation/PxErrorCallback.h"
#include "common/PxPhysXCommonConfig.h"

namespace physx
{

#if PX_VC
#pragma warning(disable: 4191)	//'operator/operation' : unsafe conversion from 'type of expression' to 'type required'
#endif

// CUDA toolkit definitions
// Update the definitions when the Cuda toolkit changes
// Refer to https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
#define MIN_CUDA_VERSION			12000	// Use Cuda toolkit 12.0 and above
#define NV_DRIVER_MAJOR_VERSION		527
#define NV_DRIVER_MINOR_VERSION		41
#define MIN_SM_MAJOR_VERSION		7
#define MIN_SM_MINOR_VERSION		0

#define USE_DEFAULT_CUDA_STREAM		0
#define FORCE_LAUNCH_SYNCHRONOUS	0
//PX_STOMP_ALLOCATED_MEMORY is defined in common/PxPhysXCommonConfig.h


#if PX_DEBUG
#include "PxgMemoryTracker.h"
static MemTracker mMemTracker;
#endif

PxCudaContext* createCudaContext(hipDevice_t device, PxDeviceAllocatorCallback* callback, bool launchSynchronous);

class CudaCtxMgr : public PxCudaContextManager, public PxUserAllocated
{
public:
	CudaCtxMgr(const PxCudaContextManagerDesc& desc, PxErrorCallback& errorCallback, bool launchSynchronous);
	virtual ~CudaCtxMgr();

	bool                    safeDelayImport(PxErrorCallback& errorCallback);

	virtual void            acquireContext() PX_OVERRIDE;
	virtual void            releaseContext() PX_OVERRIDE;
	virtual bool            tryAcquireContext() PX_OVERRIDE;

	/* All these methods can be called without acquiring the context */
	virtual bool            contextIsValid() const PX_OVERRIDE;
	virtual bool            supportsArchSM10() const PX_OVERRIDE;  // G80
	virtual bool            supportsArchSM11() const PX_OVERRIDE;  // G92
	virtual bool            supportsArchSM12() const PX_OVERRIDE;
	virtual bool            supportsArchSM13() const PX_OVERRIDE;  // GT200
	virtual bool            supportsArchSM20() const PX_OVERRIDE;  // GF100
	virtual bool            supportsArchSM30() const PX_OVERRIDE;  // GK100
	virtual bool            supportsArchSM35() const PX_OVERRIDE;  // GK110
	virtual bool            supportsArchSM50() const PX_OVERRIDE;  // GM100
	virtual bool            supportsArchSM52() const PX_OVERRIDE;  // GM200
	virtual bool            supportsArchSM60() const PX_OVERRIDE;  // GP100
	virtual bool            isIntegrated() const PX_OVERRIDE;      // true if GPU is integrated (MCP) part
	virtual bool            canMapHostMemory() const PX_OVERRIDE;  // true if GPU map host memory to GPU
	virtual int             getDriverVersion() const PX_OVERRIDE;
	virtual size_t          getDeviceTotalMemBytes() const PX_OVERRIDE;
	virtual int				getMultiprocessorCount() const PX_OVERRIDE;
	virtual int             getSharedMemPerBlock() const PX_OVERRIDE;
	virtual int             getSharedMemPerMultiprocessor() const PX_OVERRIDE;
	virtual unsigned int	getMaxThreadsPerBlock() const PX_OVERRIDE;
	virtual unsigned int	getClockRate() const PX_OVERRIDE;

	virtual const char*     getDeviceName() const PX_OVERRIDE;
	virtual hipDevice_t		getDevice() const PX_OVERRIDE;

	virtual void			setUsingConcurrentStreams(bool) PX_OVERRIDE;
	virtual bool			getUsingConcurrentStreams() const PX_OVERRIDE;

	virtual void            getDeviceMemoryInfo(size_t& free, size_t& total) const PX_OVERRIDE;

	virtual void            release() PX_OVERRIDE;

	virtual hipCtx_t		getContext() PX_OVERRIDE { return mCtx; }

	virtual PxCudaContext*  getCudaContext() PX_OVERRIDE { return mCudaCtx; }

	hipModule_t* getCuModules() PX_OVERRIDE { return mCuModules.begin(); }

	virtual hipDeviceptr_t     getMappedDevicePtr(void* pinnedHostBuffer) PX_OVERRIDE;

protected:
	virtual void* allocDeviceBufferInternal(PxU64 numBytes, const char* filename, PxI32 line) PX_OVERRIDE;
	virtual void* allocPinnedHostBufferInternal(PxU64 numBytes, const char* filename, PxI32 line) PX_OVERRIDE;

	virtual void freeDeviceBufferInternal(void* deviceBuffer) PX_OVERRIDE;
	virtual void freePinnedHostBufferInternal(void* pinnedHostBuffer) PX_OVERRIDE;

	virtual void clearDeviceBufferAsyncInternal(void* deviceBuffer, PxU32 numBytes, hipStream_t stream, PxI32 value) PX_OVERRIDE;

	virtual void copyDToHAsyncInternal(void* hostBuffer, const void* deviceBuffer, PxU32 numBytes, hipStream_t stream) PX_OVERRIDE;
	virtual void copyHToDAsyncInternal(void* deviceBuffer, const void* hostBuffer, PxU32 numBytes, hipStream_t stream) PX_OVERRIDE;
	virtual void copyDToDAsyncInternal(void* dstDeviceBuffer, const void* srcDeviceBuffer, PxU32 numBytes, hipStream_t stream) PX_OVERRIDE;

	virtual void copyDToHInternal(void* hostBuffer, const void* deviceBuffer, PxU32 numBytes) PX_OVERRIDE;
	virtual void copyHToDInternal(void* deviceBuffer, const void* hostBuffer, PxU32 numBytes) PX_OVERRIDE;

	virtual void memsetD8AsyncInternal(void* dstDeviceBuffer, const PxU8& value, PxU32 numBytes, hipStream_t stream) PX_OVERRIDE;
	virtual void memsetD32AsyncInternal(void* dstDeviceBuffer, const PxU32& value, PxU32 numIntegers, hipStream_t stream) PX_OVERRIDE;

private:

	PxArray<hipModule_t>	mCuModules;

	bool            mIsValid;
	bool			mOwnContext;
	hipDevice_t        mDevHandle;
	hipCtx_t       mCtx;
	PxCudaContext*	mCudaCtx;

	/* Cached device attributes, so threads can query w/o context */
	int             mComputeCapMajor;
	int             mComputeCapMinor;
	int				mIsIntegrated;
	int				mCanMapHost;
	int				mDriverVersion;
	size_t			mTotalMemBytes;
	int				mMultiprocessorCount;
	int				mMaxThreadsPerBlock;
	char			mDeviceName[128];
	int				mSharedMemPerBlock;
	int				mSharedMemPerMultiprocessor;
	int				mClockRate;
	bool			mUsingConcurrentStreams;
	uint32_t		mContextRefCountTls;
#if PX_DEBUG
	volatile PxI32 mPushPopCount;
#endif
};

hipDeviceptr_t CudaCtxMgr::getMappedDevicePtr(void* pinnedHostBuffer)
{
	hipDeviceptr_t dPtr = 0;
	PxCUresult result = getCudaContext()->memHostGetDevicePointer(&dPtr, pinnedHostBuffer, 0);
	if (result != hipSuccess)
		PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "Getting mapped device pointer failed with error code %i!\n", PxI32(result));
	return dPtr;
}


void* CudaCtxMgr::allocDeviceBufferInternal(PxU64 numBytes, const char* filename, PxI32 lineNumber)
{
	numBytes = PxMax(PxU64(1u), numBytes);
	PxScopedCudaLock lock(*this);
	hipDeviceptr_t ptr;
	PxCUresult result = getCudaContext()->memAlloc(&ptr, numBytes);
	if (result != hipSuccess)
		PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "Mem allocation failed with error code %i!\n", PxI32(result));
	void* deviceBuffer = reinterpret_cast<void*>(ptr);
#if PX_DEBUG
	if (deviceBuffer)
		mMemTracker.registerMemory(deviceBuffer, true, numBytes, filename, lineNumber);
#else
	PX_UNUSED(filename);
	PX_UNUSED(lineNumber);
#endif
	return deviceBuffer;
}
void* CudaCtxMgr::allocPinnedHostBufferInternal(PxU64 numBytes, const char* filename, PxI32 lineNumber)
{
	numBytes = PxMax(PxU64(1u), numBytes);
	PxScopedCudaLock lock(*this);
	void* pinnedHostBuffer;
	const unsigned int cuMemhostallocDevicemap = 0x02;
	const unsigned int cuMemhostallocPortable = 0x01;
	PxCUresult result = getCudaContext()->memHostAlloc(&pinnedHostBuffer, numBytes, cuMemhostallocDevicemap | cuMemhostallocPortable);
	if (result != hipSuccess)
		PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "Mem allocation failed with error code %i!\n", PxI32(result));

#if PX_DEBUG
	mMemTracker.registerMemory(pinnedHostBuffer, false, numBytes, filename, lineNumber);
#else
	PX_UNUSED(filename);
	PX_UNUSED(lineNumber);
#endif
	return pinnedHostBuffer;
}

void CudaCtxMgr::freeDeviceBufferInternal(void* deviceBuffer)
{
	if (!deviceBuffer)
		return;
	PxScopedCudaLock lock(*this);
	PxCUresult result = getCudaContext()->memFree(hipDeviceptr_t(deviceBuffer));
	if (result != hipSuccess)
		PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "Mem free failed with error code %i!\n", PxI32(result));
#if PX_DEBUG
	mMemTracker.unregisterMemory(deviceBuffer, true);
#endif
}
void CudaCtxMgr::freePinnedHostBufferInternal(void* pinnedHostBuffer)
{
	if (!pinnedHostBuffer)
		return;
	PxScopedCudaLock lock(*this);
	PxCUresult result = getCudaContext()->memFreeHost(pinnedHostBuffer);
	if (result != hipSuccess)
		PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "Mem free failed with error code %i!\n", PxI32(result));
#if PX_DEBUG
	mMemTracker.unregisterMemory(pinnedHostBuffer, false);
#endif
}

void CudaCtxMgr::clearDeviceBufferAsyncInternal(void* deviceBuffer, PxU32 numBytes, hipStream_t stream, PxI32 value)
{
	if (!deviceBuffer)
		return;
	PxScopedCudaLock lock(*this);
	PX_ASSERT(numBytes % 4 == 0);
	PxCUresult result = getCudaContext()->memsetD32Async(hipDeviceptr_t(deviceBuffer), value, numBytes >> 2, stream);
	if (result != hipSuccess)
		PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "Mem set failed with error code %i!\n", PxI32(result));
}

void CudaCtxMgr::copyDToHAsyncInternal(void* hostBuffer, const void* deviceBuffer, PxU32 numBytes, hipStream_t stream)
{
	if (!deviceBuffer || !hostBuffer)
		return;
	PxScopedCudaLock lock(*this);
	PxCUresult result = getCudaContext()->memcpyDtoHAsync(hostBuffer, hipDeviceptr_t(deviceBuffer), numBytes, stream);
	if (result != hipSuccess)
		PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "copyDtoHAsync set failed with error code %i!\n", PxI32(result));
}
void CudaCtxMgr::copyHToDAsyncInternal(void* deviceBuffer, const void* hostBuffer, PxU32 numBytes, hipStream_t stream)
{
	if (!deviceBuffer || !hostBuffer)
		return;
	PxScopedCudaLock lock(*this);
	PxCUresult result = getCudaContext()->memcpyHtoDAsync(hipDeviceptr_t(deviceBuffer), hostBuffer, numBytes, stream);
	if (result != hipSuccess)
		PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "copyHtoDAsync set failed with error code %i!\n", PxI32(result));
}
void CudaCtxMgr::copyDToDAsyncInternal(void* dstDeviceBuffer, const void* srcDeviceBuffer, PxU32 numBytes, hipStream_t stream)
{
	if (!srcDeviceBuffer || !dstDeviceBuffer)
		return;
	PxScopedCudaLock lock(*this);
	PxCUresult result = getCudaContext()->memcpyDtoDAsync(hipDeviceptr_t(dstDeviceBuffer), hipDeviceptr_t(srcDeviceBuffer), numBytes, stream);
	if (result != hipSuccess)
		PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "copyDtoDAsync set failed with error code %i!\n", PxI32(result));
}

void CudaCtxMgr::copyDToHInternal(void* hostBuffer, const void* deviceBuffer, PxU32 numBytes)
{
	if (!deviceBuffer || !hostBuffer)
		return;
	PxScopedCudaLock lock(*this);
	PxCUresult result = getCudaContext()->memcpyDtoH(hostBuffer, hipDeviceptr_t(deviceBuffer), numBytes);
	if (result != hipSuccess)
		PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "copyDtoH set failed with error code %i!\n", PxI32(result));
}
void CudaCtxMgr::copyHToDInternal(void* deviceBuffer, const void* hostBuffer, PxU32 numBytes)
{
	if (!deviceBuffer || !hostBuffer)
		return;
	PxScopedCudaLock lock(*this);
	PxCUresult result = getCudaContext()->memcpyHtoD(hipDeviceptr_t(deviceBuffer), hostBuffer, numBytes);
	if (result != hipSuccess)
		PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "copyHtoD set failed with error code %i!\n", PxI32(result));
}

void CudaCtxMgr::memsetD8AsyncInternal(void* dstDeviceBuffer, const PxU8& value, PxU32 numBytes, hipStream_t stream)
{
	if (!dstDeviceBuffer)
		return;
	PxScopedCudaLock lock(*this);
	PxCUresult result = getCudaContext()->memsetD8Async(hipDeviceptr_t(dstDeviceBuffer), value, numBytes, stream);
	if (result != hipSuccess)
		PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "Memset failed with error code %i!\n", PxI32(result));

}

void CudaCtxMgr::memsetD32AsyncInternal(void* dstDeviceBuffer, const PxU32& value, PxU32 numIntegers, hipStream_t stream)
{
	if (!dstDeviceBuffer)
		return;
	PxScopedCudaLock lock(*this);
	PxCUresult result = getCudaContext()->memsetD32Async(hipDeviceptr_t(dstDeviceBuffer), value, numIntegers, stream);
	if (result != hipSuccess)
		PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "Memset failed with error code %i!\n", PxI32(result));
}


bool CudaCtxMgr::contextIsValid() const
{
	return mIsValid;
}
bool CudaCtxMgr::supportsArchSM10() const
{
	return mIsValid;
}
bool CudaCtxMgr::supportsArchSM11() const
{
	return mIsValid && (mComputeCapMinor >= 1 || mComputeCapMajor > 1);
}
bool CudaCtxMgr::supportsArchSM12() const
{
	return mIsValid && (mComputeCapMinor >= 2 || mComputeCapMajor > 1);
}
bool CudaCtxMgr::supportsArchSM13() const
{
	return mIsValid && (mComputeCapMinor >= 3 || mComputeCapMajor > 1);
}
bool CudaCtxMgr::supportsArchSM20() const
{
	return mIsValid && mComputeCapMajor >= 2;
}
bool CudaCtxMgr::supportsArchSM30() const
{
	return mIsValid && mComputeCapMajor >= 3;
}
bool CudaCtxMgr::supportsArchSM35() const
{
	return mIsValid && ((mComputeCapMajor > 3) || (mComputeCapMajor == 3 && mComputeCapMinor >= 5));
}
bool CudaCtxMgr::supportsArchSM50() const
{
	return mIsValid && mComputeCapMajor >= 5;
}
bool CudaCtxMgr::supportsArchSM52() const
{
	return mIsValid && ((mComputeCapMajor > 5) || (mComputeCapMajor == 5 && mComputeCapMinor >= 2));
}
bool CudaCtxMgr::supportsArchSM60() const
{
	return mIsValid && mComputeCapMajor >= 6;
}

bool CudaCtxMgr::isIntegrated() const
{
	return mIsValid && mIsIntegrated;
}
bool CudaCtxMgr::canMapHostMemory() const
{
	return mIsValid && mCanMapHost;
}
int  CudaCtxMgr::getDriverVersion() const
{
	return mDriverVersion;
}
size_t  CudaCtxMgr::getDeviceTotalMemBytes() const
{
	return mTotalMemBytes;
}
int	CudaCtxMgr::getMultiprocessorCount() const
{
	return mMultiprocessorCount;
}
int CudaCtxMgr::getSharedMemPerBlock() const
{
	return mSharedMemPerBlock;
}
int CudaCtxMgr::getSharedMemPerMultiprocessor() const
{
	return mSharedMemPerMultiprocessor;
}
unsigned int CudaCtxMgr::getMaxThreadsPerBlock() const
{
	return (unsigned int)mMaxThreadsPerBlock;
}
unsigned int CudaCtxMgr::getClockRate() const
{
	return (unsigned int)mClockRate;
}

const char* CudaCtxMgr::getDeviceName() const
{
	if (mIsValid)
	{
		return mDeviceName;
	}
	else
	{
		return "Invalid";
	}
}

hipDevice_t CudaCtxMgr::getDevice() const
{
	if (mIsValid)
	{
		return mDevHandle;
	}
	else
	{
		return -1;
	}
}

void CudaCtxMgr::setUsingConcurrentStreams(bool value)
{
	mUsingConcurrentStreams = value;
}

bool CudaCtxMgr::getUsingConcurrentStreams() const
{
	return mUsingConcurrentStreams;
}

void CudaCtxMgr::getDeviceMemoryInfo(size_t& free, size_t& total) const
{
	hipMemGetInfo(&free, &total);
}

#define CUT_SAFE_CALL(call)  { hipError_t ret = call;	\
		if( hipSuccess != ret ) { PX_ASSERT(0); } }

/* If a context is not provided, an ordinal must be given */
CudaCtxMgr::CudaCtxMgr(const PxCudaContextManagerDesc& desc, PxErrorCallback& errorCallback, bool launchSynchronous)
	: mOwnContext(false)
	, mCudaCtx(NULL)
	, mUsingConcurrentStreams(true)
#if PX_DEBUG
	, mPushPopCount(0)
#endif
{
	hipError_t status;
	mIsValid = false;
	mDeviceName[0] = 0;

	if (safeDelayImport(errorCallback) == false)
	{
		char buffer[256];
		physx::Pxsnprintf(buffer, 256, "NVIDIA Release %u.%u graphics driver and above is required for GPU acceleration.", NV_DRIVER_MAJOR_VERSION, NV_DRIVER_MINOR_VERSION);
		errorCallback.reportError(PxErrorCode::eDEBUG_WARNING, buffer, PX_FL);
		return;
	}

	if (desc.ctx == 0)
	{
		int flags = hipDeviceLmemResizeToMax | hipDeviceScheduleBlockingSync | hipDeviceMapHost;
		class FoundationErrorReporter : public PxErrorCallback
		{
		public:
			FoundationErrorReporter(PxErrorCallback& ec)
			: errorCallback(&ec)
			{
			}

			virtual void reportError(PxErrorCode::Enum code, const char* message, const char* file, int line) PX_OVERRIDE
			{
				errorCallback->reportError(code, message, file, line);
			}

		private:
			PxErrorCallback* errorCallback;
		} foundationErrorReporter(errorCallback);

		int devOrdinal = desc.deviceOrdinal;
		if (desc.deviceOrdinal < 0)
		{
			devOrdinal = PhysXDeviceSettings::getSuggestedCudaDeviceOrdinal(foundationErrorReporter);
		}

		if (devOrdinal < 0)
		{
			errorCallback.reportError(PxErrorCode::eDEBUG_INFO, "No PhysX capable GPU suggested.", PX_FL);
			return;
		}

		status = hipInit(0);
		if (hipSuccess != status)
		{
			char buffer[128];
			physx::Pxsnprintf(buffer, 128, "hipInit failed with error code %i", status);
			errorCallback.reportError(PxErrorCode::eDEBUG_WARNING, buffer, PX_FL);
			return;
		}

		{
			status = hipDeviceGet(&mDevHandle, devOrdinal);
			if (hipSuccess != status)
			{
				errorCallback.reportError(PxErrorCode::eDEBUG_WARNING, "hipDeviceGet failed",__FILE__,__LINE__);
				return;
			}
			
			status = hipCtxCreate(&mCtx, (unsigned int)flags, mDevHandle);
			if (hipSuccess != status)
			{
				const size_t bufferSize = 128;
				char errorMsg[bufferSize];
				physx::Pxsnprintf(errorMsg, bufferSize, "hipCtxCreate failed with error %i.", status);
				errorCallback.reportError(PxErrorCode::eDEBUG_WARNING, errorMsg, PX_FL);
				return;
			}
			mOwnContext = true;
		}
	}
	else
	{
		mCtx = *desc.ctx;
		status = hipCtxGetDevice(&mDevHandle);
		if (hipSuccess != status)
		{
			errorCallback.reportError(PxErrorCode::eDEBUG_WARNING, "hipCtxGetDevice failed",__FILE__,__LINE__);
			return;
		}
	}

	// create cuda context wrapper
	mCudaCtx = createCudaContext(mDevHandle, desc.deviceAllocator, launchSynchronous);
	
	// Verify we can at least allocate a CUDA event from this context
	hipEvent_t testEvent;
	if (hipSuccess == mCudaCtx->eventCreate(&testEvent, 0))
	{
		mCudaCtx->eventDestroy(testEvent);
	}
	else
	{
		errorCallback.reportError(PxErrorCode::eDEBUG_WARNING, "CUDA context validation failed",__FILE__,__LINE__);
		return;
	}

	status = hipDeviceGetName(mDeviceName, sizeof(mDeviceName), mDevHandle);
	if (hipSuccess != status)
	{
		errorCallback.reportError(PxErrorCode::eDEBUG_WARNING, "hipDeviceGetName failed",__FILE__,__LINE__);
		return;
	}

	hipDeviceGetAttribute(&mSharedMemPerBlock, hipDeviceAttributeMaxSharedMemoryPerBlock, mDevHandle);
	hipDeviceGetAttribute(&mSharedMemPerMultiprocessor, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, mDevHandle);
	hipDeviceGetAttribute(&mClockRate, hipDeviceAttributeClockRate, mDevHandle);
	hipDeviceGetAttribute(&mComputeCapMajor, hipDeviceAttributeComputeCapabilityMajor, mDevHandle);
	hipDeviceGetAttribute(&mComputeCapMinor, hipDeviceAttributeComputeCapabilityMinor, mDevHandle);
	hipDeviceGetAttribute(&mIsIntegrated, hipDeviceAttributeIntegrated, mDevHandle);
	hipDeviceGetAttribute(&mCanMapHost, hipDeviceAttributeCanMapHostMemory, mDevHandle);
	hipDeviceGetAttribute(&mMultiprocessorCount, hipDeviceAttributeMultiprocessorCount, mDevHandle);
	hipDeviceGetAttribute(&mMaxThreadsPerBlock, hipDeviceAttributeMaxThreadsPerBlock, mDevHandle);

	status = hipDeviceTotalMem((size_t*)&mTotalMemBytes, mDevHandle);
	if (hipSuccess != status)
	{
		errorCallback.reportError(PxErrorCode::eDEBUG_WARNING, "hipDeviceTotalMem failed",__FILE__,__LINE__);
		return;
	}

	// minimum compute capability is MIN_SM_MAJOR_VERSION.MIN_SM_MINOR_VERSION
	if ((mComputeCapMajor < MIN_SM_MAJOR_VERSION)	||
		(mComputeCapMajor == MIN_SM_MAJOR_VERSION && mComputeCapMinor < MIN_SM_MINOR_VERSION))
	{
		char buffer[256];
		physx::Pxsnprintf(buffer, 256, "Minimum GPU compute capability %d.%d is required", MIN_SM_MAJOR_VERSION, MIN_SM_MINOR_VERSION);
		errorCallback.reportError(PxErrorCode::eDEBUG_WARNING,buffer,__FILE__,__LINE__);
		return;
	}

	mContextRefCountTls = PxTlsAlloc();
	mIsValid = true;

	// Formally load the CUDA modules, get hipModule_t handles
	{
		PxScopedCudaLock lock(*this);
		const PxU32 moduleTableSize = PxGpuGetCudaModuleTableSize();
		void** moduleTable = PxGpuGetCudaModuleTable();
		mCuModules.resize(moduleTableSize, NULL);
		for (PxU32 i = 0 ; i < moduleTableSize ; ++i)
		{
			hipError_t ret = hipErrorUnknown;

			// Make sure that moduleTable[i] is not null
			if (moduleTable[i])
			{
				ret = mCudaCtx->moduleLoadDataEx(&mCuModules[i], moduleTable[i], 0, NULL, NULL);
			}

			if (ret != hipSuccess && ret != hipErrorNoBinaryForGpu)
			{
				const size_t bufferSize = 256;
				char errorMsg[bufferSize];
				physx::Pxsnprintf(errorMsg, bufferSize, "Failed to load CUDA module data. Cuda error code %i.\n", ret);

				PxGetErrorCallback()->reportError(PxErrorCode::eINTERNAL_ERROR, errorMsg, PX_FL);
				mCuModules[i] = NULL;
			}
		}
	}
}

/* Some driver version mismatches can cause delay import crashes.  Load NVCUDA.dll
 * manually, verify its version number, then allow delay importing to bind all the
 * APIs.
 */
bool CudaCtxMgr::safeDelayImport(PxErrorCallback& errorCallback)
{
#if PX_WIN32 || PX_WIN64
	HMODULE hCudaDriver = LoadLibrary("nvcuda.dll");
#elif PX_LINUX
	void*	hCudaDriver = dlopen("libcuda.so.1", RTLD_NOW);
#endif
	if (!hCudaDriver)
	{
		errorCallback.reportError(PxErrorCode::eINTERNAL_ERROR, "nvcuda.dll not found or could not be loaded.", PX_FL);
		return false;
	}

	typedef hipError_t(CUDAAPI * pfnCuDriverGetVersion_t)(int*);
	pfnCuDriverGetVersion_t pfnCuDriverGetVersion = (pfnCuDriverGetVersion_t) GetProcAddress(hCudaDriver, "hipDriverGetVersion");
	if (!pfnCuDriverGetVersion)
	{
		errorCallback.reportError(PxErrorCode::eINTERNAL_ERROR, "hipDriverGetVersion missing in nvcuda.dll.", PX_FL);
		return false;
	}

	#if PX_A64
		hipError_t status = hipDriverGetVersion(&mDriverVersion);
	#else
		hipError_t status = pfnCuDriverGetVersion(&mDriverVersion);
	#endif

	if (status != hipSuccess)
	{
		errorCallback.reportError(PxErrorCode::eINTERNAL_ERROR, "Retrieving CUDA driver version failed.", PX_FL);
		return false;
	}

	// Check that the Cuda toolkit used meets the minimum version
	// If the Cuda toolkit has changed, refer to https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
	// Make the necessary changes to Cuda toolkit definitions
	PX_COMPILE_TIME_ASSERT(CUDA_VERSION >= MIN_CUDA_VERSION);

	// Check whether Cuda driver version meets the min requirement
	if (mDriverVersion < MIN_CUDA_VERSION)
	{
		char buffer[256];
		physx::Pxsnprintf(buffer, 256, "CUDA driver version is %u, expected driver version is at least %u.", mDriverVersion, MIN_CUDA_VERSION);
		errorCallback.reportError(PxErrorCode::eINTERNAL_ERROR, buffer, __FILE__,__LINE__);
		return false;
	}

	/* Now trigger delay import and API binding */
	status = hipDriverGetVersion(&mDriverVersion);
	if (status != hipSuccess)
	{
		errorCallback.reportError(PxErrorCode::eINTERNAL_ERROR, "Failed to bind CUDA API.", PX_FL);
		return false;
	}

	/* Not strictly necessary, but good practice */
#if PX_WIN32 | PX_WIN64
	FreeLibrary(hCudaDriver);
#elif PX_LINUX
	dlclose(hCudaDriver);
#endif

	return true;
}

void CudaCtxMgr::release()
{
	PX_DELETE_THIS;
}

CudaCtxMgr::~CudaCtxMgr()
{
	if (mCudaCtx)
	{
		// unload CUDA modules
		{
			PxScopedCudaLock lock(*this);
			for(PxU32 i = 0; i < mCuModules.size(); i++)
			{
				hipError_t ret = mCudaCtx->moduleUnload(mCuModules[i]);
				if(ret != hipSuccess)
				{
					char msg[128];
					physx::Pxsnprintf(msg, 128, "Failed to unload CUDA module data, returned %i.", ret);
					PxGetErrorCallback()->reportError(PxErrorCode::eINTERNAL_ERROR, msg, PX_FL);
				}
			}
		}

		mCudaCtx->release();
		mCudaCtx = NULL;
	}

	if (mOwnContext)
	{
		CUT_SAFE_CALL(hipCtxDestroy(mCtx));
	}

	PxTlsFree(mContextRefCountTls);

#if PX_DEBUG
	PX_ASSERT(mPushPopCount == 0);
#endif
}

void CudaCtxMgr::acquireContext()
{
	bool result = tryAcquireContext();
	PX_ASSERT(result);
	PX_UNUSED(result);
}

bool CudaCtxMgr::tryAcquireContext()
{
	// AD: we directly store the counter in the per-thread value (instead of using a pointer-to-value.)
	// Using size_t because we have a pointer's width to play with, so the type will potentially depend on the platform.
	// All the values are initialized to NULL at PxTlsAlloc() and for any newly created thread it will be NULL as well.
	// So even if a thread hits this code for the first time, we know it's zero, and then we start by placing the correct refcount
	// below in the set call.
	size_t refCount = PxTlsGetValue(mContextRefCountTls);

	hipError_t result = hipSuccess;

#if PX_DEBUG
	result = hipCtxPushCurrent(mCtx);
	PxAtomicIncrement(&mPushPopCount);
#else
	if (refCount == 0)
	{
		result = hipCtxPushCurrent(mCtx);
	}
#endif
	PxTlsSetValue(mContextRefCountTls, ++refCount);

	return result == hipSuccess;
}

void CudaCtxMgr::releaseContext()
{
	size_t refCount = PxTlsGetValue(mContextRefCountTls);

#if PX_DEBUG
	hipCtx_t ctx = 0;
	CUT_SAFE_CALL(hipCtxPopCurrent(&ctx));
	PxAtomicDecrement(&mPushPopCount);
#else
	if (--refCount == 0)
	{
		hipCtx_t ctx = 0;
		CUT_SAFE_CALL(hipCtxPopCurrent(&ctx));
	}
#endif
	PxTlsSetValue(mContextRefCountTls, refCount);
}

class CudaCtx : public PxCudaContext, public PxUserAllocated
{
private:
	hipError_t mLastResult;
	bool mLaunchSynchronous;
	bool mIsInAbortMode;

public:
	CudaCtx(PxDeviceAllocatorCallback* callback, bool launchSynchronous);
	~CudaCtx();

	// PxCudaContext
	void		release()	PX_OVERRIDE PX_FINAL;
	PxCUresult	memAlloc(hipDeviceptr_t* dptr, size_t bytesize)	PX_OVERRIDE PX_FINAL;
	PxCUresult	memFree(hipDeviceptr_t dptr)	PX_OVERRIDE PX_FINAL;
	PxCUresult	memHostAlloc(void** pp, size_t bytesize, unsigned int Flags)	PX_OVERRIDE PX_FINAL;
	PxCUresult	memFreeHost(void* p)	PX_OVERRIDE PX_FINAL;
	PxCUresult	memHostGetDevicePointer(hipDeviceptr_t* pdptr, void* p, unsigned int Flags)	PX_OVERRIDE PX_FINAL;
	PxCUresult	moduleLoadDataEx(hipModule_t* module, const void* image, unsigned int numOptions, PxCUjit_option* options, void** optionValues)	PX_OVERRIDE PX_FINAL;
	PxCUresult	moduleGetFunction(hipFunction_t* hfunc, hipModule_t hmod, const char* name)	PX_OVERRIDE PX_FINAL;
	PxCUresult	moduleUnload(hipModule_t hmod)	PX_OVERRIDE PX_FINAL;
	PxCUresult	streamCreate(hipStream_t* phStream, unsigned int Flags)	PX_OVERRIDE PX_FINAL;
	PxCUresult	streamCreateWithPriority(hipStream_t* phStream, unsigned int flags, int priority)	PX_OVERRIDE PX_FINAL;
	PxCUresult	streamFlush(hipStream_t hStream)	PX_OVERRIDE PX_FINAL;
	PxCUresult	streamWaitEvent(hipStream_t hStream, hipEvent_t hEvent, unsigned int Flags)	PX_OVERRIDE PX_FINAL;
	PxCUresult	streamWaitEvent(hipStream_t hStream, hipEvent_t hEvent)	PX_OVERRIDE PX_FINAL;
	PxCUresult	streamDestroy(hipStream_t hStream)	PX_OVERRIDE PX_FINAL;
	PxCUresult	streamSynchronize(hipStream_t hStream)	PX_OVERRIDE PX_FINAL;
	PxCUresult	eventCreate(hipEvent_t* phEvent, unsigned int Flags)	PX_OVERRIDE PX_FINAL;
	PxCUresult	eventRecord(hipEvent_t hEvent, hipStream_t hStream)	PX_OVERRIDE PX_FINAL;
	PxCUresult	eventQuery(hipEvent_t hEvent)	PX_OVERRIDE PX_FINAL;
	PxCUresult	eventSynchronize(hipEvent_t hEvent)	PX_OVERRIDE PX_FINAL;
	PxCUresult	eventDestroy(hipEvent_t hEvent)	PX_OVERRIDE PX_FINAL;

	PxCUresult launchKernel(
		hipFunction_t f,
		PxU32 gridDimX, PxU32 gridDimY, PxU32 gridDimZ,
		PxU32 blockDimX, PxU32 blockDimY, PxU32 blockDimZ,
		PxU32 sharedMemBytes,
		hipStream_t hStream,
		PxCudaKernelParam* kernelParams,
		size_t kernelParamsSizeInBytes,
		void** extra,
		const char* file,
		int line
	)	PX_OVERRIDE PX_FINAL;

	PxCUresult launchKernel(
		hipFunction_t f,
		PxU32 gridDimX, PxU32 gridDimY, PxU32 gridDimZ,
		PxU32 blockDimX, PxU32 blockDimY, PxU32 blockDimZ,
		PxU32 sharedMemBytes,
		hipStream_t hStream,
		void** kernelParams,
		void** extra,
		const char* file,
		int line
	)	PX_OVERRIDE PX_FINAL;
	
	PxCUresult memcpyDtoH(void* dstHost, hipDeviceptr_t srcDevice, size_t ByteCount)	PX_OVERRIDE PX_FINAL;
	PxCUresult memcpyDtoHAsync(void* dstHost, hipDeviceptr_t srcDevice, size_t ByteCount, hipStream_t hStream)	PX_OVERRIDE PX_FINAL;
	PxCUresult memcpyHtoD(hipDeviceptr_t dstDevice, const void* srcHost, size_t ByteCount)	PX_OVERRIDE PX_FINAL;
	PxCUresult memcpyHtoDAsync(hipDeviceptr_t dstDevice, const void* srcHost, size_t ByteCount, hipStream_t hStream)	PX_OVERRIDE PX_FINAL;
	PxCUresult memcpyDtoD(hipDeviceptr_t dstDevice, hipDeviceptr_t srcDevice, size_t ByteCount)	PX_OVERRIDE PX_FINAL;
	PxCUresult memcpyDtoDAsync(hipDeviceptr_t dstDevice, hipDeviceptr_t srcDevice, size_t ByteCount, hipStream_t hStream)	PX_OVERRIDE PX_FINAL;
	PxCUresult memcpyPeerAsync(hipDeviceptr_t dstDevice, hipCtx_t dstContext, hipDeviceptr_t srcDevice, hipCtx_t srcContext, size_t ByteCount, hipStream_t hStream)	PX_OVERRIDE PX_FINAL;
	PxCUresult memsetD32Async(hipDeviceptr_t dstDevice, unsigned int ui, size_t N, hipStream_t hStream)	PX_OVERRIDE PX_FINAL;
	PxCUresult memsetD8Async(hipDeviceptr_t dstDevice, unsigned char uc, size_t N, hipStream_t hStream)	PX_OVERRIDE PX_FINAL;
	PxCUresult memsetD32(hipDeviceptr_t dstDevice, unsigned int ui, size_t N)	PX_OVERRIDE PX_FINAL;
	PxCUresult memsetD16(hipDeviceptr_t dstDevice, unsigned short uh, size_t N)	PX_OVERRIDE PX_FINAL;
	PxCUresult memsetD8(hipDeviceptr_t dstDevice, unsigned char uc, size_t N)	PX_OVERRIDE PX_FINAL;
	PxCUresult getLastError()	PX_OVERRIDE PX_FINAL	{ return isInAbortMode() ? hipErrorOutOfMemory : mLastResult; }

	void setAbortMode(bool abort) PX_OVERRIDE PX_FINAL;
	bool isInAbortMode() PX_OVERRIDE PX_FINAL { return mIsInAbortMode; }

	//~PxCudaContext
};

CudaCtx::CudaCtx(PxDeviceAllocatorCallback* callback, bool launchSynchronous)
{
	mLastResult = hipSuccess;
	mAllocatorCallback = callback;
	mIsInAbortMode = false;
#if FORCE_LAUNCH_SYNCHRONOUS
	PX_UNUSED(launchSynchronous);
	mLaunchSynchronous = true;
#else
	mLaunchSynchronous = launchSynchronous;
#endif
}

CudaCtx::~CudaCtx()
{

}

void CudaCtx::release()
{
	PX_DELETE_THIS;
}

PxCUresult CudaCtx::memAlloc(hipDeviceptr_t *dptr, size_t bytesize)
{
	if (mIsInAbortMode)
	{
		*dptr = NULL;
		return mLastResult;
	}

	mLastResult = hipMalloc(dptr, bytesize);
#if PX_STOMP_ALLOCATED_MEMORY
	if(*dptr && bytesize > 0)
	{
		hipCtxSynchronize();
		PxCUresult result = memsetD8(*dptr, PxU8(0xcd), bytesize);
		PX_ASSERT(result == hipSuccess);
		PX_UNUSED(result);
		hipCtxSynchronize();
	}
#endif
	return mLastResult;
}

PxCUresult CudaCtx::memFree(hipDeviceptr_t dptr)
{
	if ((void*)dptr == NULL)
		return mLastResult;

 	return hipFree(dptr);
}

PxCUresult CudaCtx::memHostAlloc(void** pp, size_t bytesize, unsigned int Flags)
{
	hipError_t result = hipHostAlloc(pp, bytesize, Flags);
#if PX_STOMP_ALLOCATED_MEMORY
	if(*pp != NULL && bytesize > 0)
	{
		PxMemSet(*pp, PxI32(0xcd), PxU32(bytesize));
	}
#endif
	return result;
}

PxCUresult CudaCtx::memFreeHost(void* p)
{
	return hipHostFree(p);
}

PxCUresult CudaCtx::memHostGetDevicePointer(hipDeviceptr_t* pdptr, void* p, unsigned int Flags)
{
	if (!p)
	{
		*pdptr = reinterpret_cast<hipDeviceptr_t>(p);
		return hipSuccess;
	}
	return hipHostGetDevicePointer(pdptr, p, Flags);
}

PxCUresult CudaCtx::moduleLoadDataEx(hipModule_t* module, const void* image, unsigned int numOptions, PxCUjit_option* options, void** optionValues)
{
	return hipModuleLoadDataEx(module, image, numOptions, (hipJitOption*)options, optionValues);
}

PxCUresult CudaCtx::moduleGetFunction(hipFunction_t* hfunc, hipModule_t hmod, const char* name)
{
	return hipModuleGetFunction(hfunc, hmod, name);
}

PxCUresult CudaCtx::moduleUnload(hipModule_t hmod)
{
	return hipModuleUnload(hmod);
}

PxCUresult CudaCtx::streamCreate(hipStream_t* phStream, unsigned int Flags)
{
	if (mIsInAbortMode)
	{
		*phStream = NULL;
		return mLastResult;
	}

#if !USE_DEFAULT_CUDA_STREAM
	mLastResult = hipStreamCreateWithFlags(phStream, Flags);
#else
	PX_UNUSED(Flags);
	*phStream = hipStream_t(hipStreamDefault);
	mLastResult = hipSuccess;
#endif

	return mLastResult;
}

PxCUresult CudaCtx::streamCreateWithPriority(hipStream_t* phStream, unsigned int flags, int priority)
{
	if (mIsInAbortMode)
	{
		*phStream = NULL;
		return mLastResult;
	}

#if !USE_DEFAULT_CUDA_STREAM
	mLastResult = hipStreamCreateWithPriority(phStream, flags, priority);
#else
	PX_UNUSED(flags);
	PX_UNUSED(priority);
	*phStream = hipStream_t(hipStreamDefault);
	mLastResult = hipSuccess;
#endif

	return mLastResult;
}

PxCUresult CudaCtx::streamFlush(hipStream_t hStream)
{
	if (mIsInAbortMode)
		return mLastResult;

	// AD: don't remember the error, because this can return hipErrorNotReady which is not really an error.
	// We just misuse streamquery to push the buffer anyway.
	return hipStreamQuery(hStream);
}

PxCUresult CudaCtx::streamWaitEvent(hipStream_t hStream, hipEvent_t hEvent, unsigned int Flags)
{
	if (mIsInAbortMode)
		return mLastResult;

	mLastResult = hipStreamWaitEvent(hStream, hEvent, Flags);
	return mLastResult;
}

PxCUresult CudaCtx::streamWaitEvent(hipStream_t hStream, hipEvent_t hEvent)
{
	return streamWaitEvent(hStream, hEvent, 0);
}

PxCUresult CudaCtx::streamDestroy(hipStream_t hStream)
{
	PX_UNUSED(hStream);
#if !USE_DEFAULT_CUDA_STREAM
	if (hStream == NULL)
		return mLastResult;
	return hipStreamDestroy(hStream);
#else
	return hipSuccess;
#endif
}

PxCUresult CudaCtx::streamSynchronize(hipStream_t hStream)
{
	if (mIsInAbortMode)
		return mLastResult;

	mLastResult = hipStreamSynchronize(hStream);
	return mLastResult;
}

PxCUresult CudaCtx::eventCreate(hipEvent_t* phEvent, unsigned int Flags)
{
	if (mIsInAbortMode)
	{
		*phEvent = NULL;
		return mLastResult;
	}

	mLastResult = hipEventCreateWithFlags(phEvent, Flags);
	return mLastResult;
}

PxCUresult CudaCtx::eventRecord(hipEvent_t hEvent, hipStream_t hStream)
{
	if (mIsInAbortMode)
		return mLastResult;

	mLastResult = hipEventRecord(hEvent, hStream);
	return mLastResult;
}

PxCUresult CudaCtx::eventQuery(hipEvent_t hEvent)
{
	if (mIsInAbortMode)
		return mLastResult;

	mLastResult = hipEventQuery(hEvent);
	return mLastResult;
}

PxCUresult CudaCtx::eventSynchronize(hipEvent_t hEvent)
{
	if (mIsInAbortMode)
		return mLastResult;

	mLastResult = hipEventSynchronize(hEvent);
	return mLastResult;
}

PxCUresult CudaCtx::eventDestroy(hipEvent_t hEvent)
{
	if (hEvent == NULL)
		return mLastResult;

	return hipEventDestroy(hEvent);
}

PxCUresult CudaCtx::launchKernel(
	hipFunction_t f,
	PxU32 gridDimX, PxU32 gridDimY, PxU32 gridDimZ,
	PxU32 blockDimX, PxU32 blockDimY, PxU32 blockDimZ,
	PxU32 sharedMemBytes,
	hipStream_t hStream,
	PxCudaKernelParam* kernelParams,
	size_t kernelParamsSizeInBytes,
	void** extra,
	const char* file,
	int line
)
{
	if (mIsInAbortMode)
		return mLastResult;

	//We allow hipErrorInvalidValue to be non-terminal error as this is sometimes  hit
	//when we launch an empty block
	if (mLastResult == hipSuccess || mLastResult == hipErrorInvalidValue)
	{
		const uint32_t kernelParamCount = (uint32_t)(kernelParamsSizeInBytes / sizeof(PxCudaKernelParam));
		PX_ALLOCA(kernelParamsLocal, void*, kernelParamCount);
		for (unsigned int paramIdx = 0u; paramIdx < kernelParamCount; paramIdx++)
		{
			kernelParamsLocal[paramIdx] = kernelParams[paramIdx].data;
		}
		mLastResult = hipModuleLaunchKernel(
			f,
			gridDimX, gridDimY, gridDimZ,
			blockDimX, blockDimY, blockDimZ,
			sharedMemBytes,
			hStream,
			kernelParamsLocal,
			extra
		);

		if (mLaunchSynchronous)
		{
			mLastResult = hipStreamSynchronize(hStream);
			if (mLastResult != hipSuccess)
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, file, line, "Launch failed!! Error: %i\n", mLastResult);
		}

		PX_ASSERT(mLastResult == hipSuccess || mLastResult == hipErrorInvalidValue);
	}

	return mLastResult;
}

PxCUresult CudaCtx::launchKernel(
	hipFunction_t f,
	PxU32 gridDimX, PxU32 gridDimY, PxU32 gridDimZ,
	PxU32 blockDimX, PxU32 blockDimY, PxU32 blockDimZ,
	PxU32 sharedMemBytes,
	hipStream_t hStream,
	void** kernelParams,
	void** extra,
	const char* file,
	int line
)
{
	if (mIsInAbortMode)
		return mLastResult;

	//We allow hipErrorInvalidValue to be non-terminal error as this is sometimes  hit
	//when we launch an empty block
	if (mLastResult == hipSuccess || mLastResult == hipErrorInvalidValue)
	{
		mLastResult = hipModuleLaunchKernel(
			f,
			gridDimX, gridDimY, gridDimZ,
			blockDimX, blockDimY, blockDimZ,
			sharedMemBytes,
			hStream,
			kernelParams,
			extra
		);

		if (mLaunchSynchronous)
		{
			mLastResult = hipStreamSynchronize(hStream);
			if (mLastResult != hipSuccess)
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, file, line, "Launch failed!! Error: %i\n", mLastResult);
		}

		PX_ASSERT(mLastResult == hipSuccess || mLastResult == hipErrorInvalidValue);
	}

	return mLastResult;
}

PxCUresult CudaCtx::memcpyDtoH(void* dstHost, hipDeviceptr_t srcDevice, size_t ByteCount)
{
	if (mIsInAbortMode)
		return mLastResult;

	if (ByteCount > 0)
		mLastResult = hipMemcpyDtoH(dstHost, srcDevice, ByteCount);
	
	if (mLastResult != hipSuccess)
	{
		PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "copyDToH failed with error code %i!\n", PxI32(mLastResult));
	}

	return mLastResult;
	
}

PxCUresult CudaCtx::memcpyDtoHAsync(void* dstHost, hipDeviceptr_t srcDevice, size_t ByteCount, hipStream_t hStream)
{
	if (mIsInAbortMode)
		return mLastResult;

	if (ByteCount > 0)
	{
		mLastResult = hipMemcpyDtoHAsync(dstHost, srcDevice, ByteCount, hStream);
		if (mLaunchSynchronous)
		{
			if (mLastResult != hipSuccess)
			{
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "memcpyDtoHAsync invalid parameters!! Error: %i\n", mLastResult);
			}
			mLastResult = hipStreamSynchronize(hStream);
			if (mLastResult != hipSuccess)
			{
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "memcpyDtoHAsync failed!! Error: %i\n", mLastResult);
			}
		}
	}
	return mLastResult;
}

PxCUresult CudaCtx::memcpyHtoD(hipDeviceptr_t dstDevice, const void* srcHost, size_t ByteCount)
{
	if (mIsInAbortMode)
		return mLastResult;

	if (ByteCount > 0)
	{
		mLastResult = hipMemcpyHtoD(dstDevice, srcHost, ByteCount);
		if (mLastResult != hipSuccess)
		{
			PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "memcpyHtoD invalid parameters!! %i\n", mLastResult);
		}
	}
	return mLastResult;
}

PxCUresult CudaCtx::memcpyHtoDAsync(hipDeviceptr_t dstDevice, const void* srcHost, size_t ByteCount, hipStream_t hStream)
{
	if (mIsInAbortMode)
		return mLastResult;

	if (ByteCount > 0)
	{
		mLastResult = hipMemcpyHtoDAsync(dstDevice, srcHost, ByteCount, hStream);
		if (mLaunchSynchronous)
		{
			if (mLastResult != hipSuccess)
			{
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "memcpyHtoDAsync invalid parameters!! Error: %i\n", mLastResult);
			}
			mLastResult = hipStreamSynchronize(hStream);
			if (mLastResult != hipSuccess)
			{
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "memcpyHtoDAsync failed!! Error: %i\n", mLastResult);
			}
		}
	}
	return mLastResult;
}

PxCUresult CudaCtx::memcpyDtoDAsync(hipDeviceptr_t dstDevice, hipDeviceptr_t srcDevice, size_t ByteCount, hipStream_t hStream)
{
	if (mIsInAbortMode)
		return mLastResult;

	if (ByteCount > 0)
	{
		mLastResult = hipMemcpyDtoDAsync(dstDevice, srcDevice, ByteCount, hStream);
		if (mLaunchSynchronous)
		{
			if (mLastResult != hipSuccess)
			{
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "memcpyDtoDAsync invalid parameters!! Error: %i\n", mLastResult);
			}
			mLastResult = hipStreamSynchronize(hStream);
			if (mLastResult != hipSuccess)
			{
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "memcpyDtoDAsync failed!! Error: %i\n", mLastResult);
			}
		}
	}
	return mLastResult;
}

PxCUresult CudaCtx::memcpyDtoD(hipDeviceptr_t dstDevice, hipDeviceptr_t srcDevice, size_t ByteCount)
{
	if (mIsInAbortMode)
		return mLastResult;

	if (ByteCount > 0)
	{
		mLastResult = hipMemcpyDtoD(dstDevice, srcDevice, ByteCount);
		// synchronize to avoid race conditions. 
		// https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html#api-sync-behavior__memcpy
		mLastResult = hipStreamSynchronize(0);
		if (mLastResult != hipSuccess)
		{
			PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "memcpyDtoD invalid parameters!! Error: %i\n", mLastResult);
		}
	}
	return mLastResult;
}

PxCUresult CudaCtx::memcpyPeerAsync(hipDeviceptr_t dstDevice, hipCtx_t dstContext, hipDeviceptr_t srcDevice, hipCtx_t srcContext, size_t ByteCount, hipStream_t hStream)
{
	if (mIsInAbortMode)
		return mLastResult;

	return cuMemcpyPeerAsync(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream);
}

PxCUresult CudaCtx::memsetD32Async(hipDeviceptr_t dstDevice, unsigned int ui, size_t N, hipStream_t hStream)
{
	if (mIsInAbortMode)
		return mLastResult;

	if (N > 0)
	{
		mLastResult = hipMemsetD32Async(dstDevice, ui, N, hStream);
		if (mLaunchSynchronous)
		{
			PX_ASSERT(mLastResult == hipSuccess);
			if (mLastResult != hipSuccess)
			{
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "memsetD32Async invalid parameters!! Error: %i\n", mLastResult);
			}
			mLastResult = hipStreamSynchronize(hStream);
			if (mLastResult != hipSuccess)
			{
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "memsetD32Async failed!! Error: %i\n", mLastResult);
			}
		}
	}
	return mLastResult;
}

PxCUresult CudaCtx::memsetD8Async(hipDeviceptr_t dstDevice, unsigned char uc, size_t N, hipStream_t hStream)
{
	if (mIsInAbortMode)
		return mLastResult;

	if (N > 0)
	{
		mLastResult = hipMemsetD8Async(dstDevice, uc, N, hStream);
		if (mLaunchSynchronous)
		{
			if (mLastResult!= hipSuccess)
			{
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "hipMemsetD8Async invalid parameters!! Error: %i\n", mLastResult);
			}
			mLastResult = hipStreamSynchronize(hStream);
			if (mLastResult != hipSuccess)
			{
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "hipMemsetD8Async failed!! Error: %i\n", mLastResult);
			}
		}
	}
	return mLastResult;
}

PxCUresult CudaCtx::memsetD32(hipDeviceptr_t dstDevice, unsigned int ui, size_t N)
{
	if (mIsInAbortMode)
		return mLastResult;

	if (N > 0)
	{
		mLastResult = hipMemsetD32(dstDevice, ui, N);
		if (mLastResult != hipSuccess)
		{
			PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "memsetD32 failed!! Error: %i\n", mLastResult);
			return mLastResult;
		}

		// synchronize to avoid race conditions.
		// https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html#api-sync-behavior__memset
		mLastResult = hipStreamSynchronize(0);
		if (mLastResult != hipSuccess)
		{
			PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "memsetD32 failed!! Error: %i\n", mLastResult);
		}
	}
	return mLastResult;
}

PxCUresult CudaCtx::memsetD16(hipDeviceptr_t dstDevice, unsigned short uh, size_t N)
{
	if (mIsInAbortMode)
		return mLastResult;

	if (N > 0)
	{
		hipMemsetD16(dstDevice, uh, N);
		// synchronize to avoid race conditions. 
		// https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html#api-sync-behavior__memset
		mLastResult = hipStreamSynchronize(0);
		if (mLastResult != hipSuccess)
		{
			PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "memsetD16 failed!! Error: %i\n", mLastResult);
		}
	}
	return mLastResult;
}

PxCUresult CudaCtx::memsetD8(hipDeviceptr_t dstDevice, unsigned char uc, size_t N)
{
	if (mIsInAbortMode)
		return mLastResult;

	if (N > 0)
	{
		hipMemsetD8(dstDevice, uc, N);
		// synchronize to avoid race conditions. 
		// https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html#api-sync-behavior__memset
		mLastResult = hipStreamSynchronize(0);
		if (mLastResult != hipSuccess)
		{
			PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "memsetD8 failed!! Error: %i\n", mLastResult);
		}
	}
	return mLastResult;
}

void CudaCtx::setAbortMode(bool abort)
{
	mIsInAbortMode = abort;
	
	if ((abort == false) && (mLastResult == hipErrorOutOfMemory))
	{	
		mLastResult = hipSuccess;
	}
}

PxCudaContext* createCudaContext(hipDevice_t device, PxDeviceAllocatorCallback* callback, bool launchSynchronous)
{
	PX_UNUSED(device);
	return PX_NEW(CudaCtx)(callback, launchSynchronous);
}

#if PX_SUPPORT_GPU_PHYSX

PxCudaContextManager* createCudaContextManager(const PxCudaContextManagerDesc& desc, PxErrorCallback& errorCallback, bool launchSynchronous)
{
	return PX_NEW(CudaCtxMgr)(desc, errorCallback, launchSynchronous);
}

#endif

} // end physx namespace


