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

#ifndef PX_CUDA_TYPES_H
#define PX_CUDA_TYPES_H

//type definitions to avoid forced inclusion of hip/hip_runtime.h
//if hip/hip_runtime.h is needed anyway, please include it before PxCudaContextManager.h, PxCudaContext.h or PxCudaTypes.h

#include "foundation/PxPreprocessor.h"

#if PX_SUPPORT_GPU_PHYSX
#ifndef CUDA_VERSION

#include "foundation/PxSimpleTypes.h"

#if PX_CLANG
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++98-compat-pedantic"
#endif

#if PX_P64_FAMILY
typedef unsigned long long hipDeviceptr_t;
#else
typedef unsigned int hipDeviceptr_t;
#endif

#if PX_CLANG
#pragma clang diagnostic pop
#endif

typedef int hipDevice_t;

typedef struct ihipCtx_t* hipCtx_t;
typedef struct ihipModule_t* hipModule_t;
typedef struct ihipModuleSymbol_t* hipFunction_t;
typedef struct ihipStream_t* hipStream_t;
typedef struct ihipEvent_t* hipEvent_t;
typedef struct hipGraphicsResource* hipGraphicsResource_t;

#endif

#else
typedef struct ihipStream_t* hipStream_t; // We declare some callbacks taking hipStream_t as an argument even when building with PX_SUPPORT_GPU_PHYSX = 0.
typedef struct ihipEvent_t* hipEvent_t;
#endif // PX_SUPPORT_GPU_PHYSX
#endif

