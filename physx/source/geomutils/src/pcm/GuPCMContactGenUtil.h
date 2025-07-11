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

#ifndef GU_PCM_CONTACT_GEN_UTIL_H
#define GU_PCM_CONTACT_GEN_UTIL_H

#include "foundation/PxVecMath.h"
#include "geomutils/PxContactBuffer.h"
#include "GuShapeConvex.h"
#include "GuVecCapsule.h"
#include "GuConvexSupportTable.h"

//The smallest epsilon we will permit (scaled by PxTolerancesScale.length)
#define	PCM_WITNESS_POINT_LOWER_EPS		1e-2f
//The largest epsilon we will permit (scaled by PxTolerancesScale.length)
#define	PCM_WITNESS_POINT_UPPER_EPS		5e-2f

namespace physx
{
namespace Gu
{
	enum FeatureStatus
	{
		POLYDATA0,
		POLYDATA1,
		EDGE
	};

	bool contains(aos::Vec3V* verts, PxU32 numVerts, const aos::Vec3VArg p, const aos::Vec3VArg min, const aos::Vec3VArg max);

	PX_FORCE_INLINE aos::FloatV signed2DTriArea(const aos::Vec3VArg a, const aos::Vec3VArg b, const aos::Vec3VArg c)
	{
		using namespace aos;
		const Vec3V ca = V3Sub(a, c);
		const Vec3V cb = V3Sub(b, c);

		const FloatV t0 = FMul(V3GetX(ca), V3GetY(cb));
		const FloatV t1 = FMul(V3GetY(ca), V3GetX(cb));

		return FSub(t0, t1);
	}

	PxI32 getPolygonIndex(const Gu::PolygonalData& polyData, const SupportLocal* map, const aos::Vec3VArg normal, PxI32& polyIndex2);

	PX_FORCE_INLINE PxI32 getPolygonIndex(const Gu::PolygonalData& polyData, const SupportLocal* map, const aos::Vec3VArg normal)
	{
		using namespace aos;

		PxI32 index2;
		return getPolygonIndex(polyData, map, normal, index2);
	}

	PxU32 getWitnessPolygonIndex(	const Gu::PolygonalData& polyData, const SupportLocal* map, const aos::Vec3VArg normal,
									const aos::Vec3VArg closest, PxReal tolerance);

	PX_FORCE_INLINE void outputPCMContact(PxContactBuffer& contactBuffer, PxU32& contactCount, const aos::Vec3VArg point, const aos::Vec3VArg normal,
											const aos::FloatVArg penetration, PxU32 internalFaceIndex1 = PXC_CONTACT_NO_FACE_INDEX)
	{
		using namespace aos;

		// PT: TODO: the PCM capsule-capsule code was using this alternative version, is it better?
		// const Vec4V normalSep = V4SetW(Vec4V_From_Vec3V(normal), separation);
		// V4StoreA(normalSep, &point.normal.x);
		// Also aren't we overwriting maxImpulse with the position now? Ok to do so?
		PX_ASSERT(contactCount < PxContactBuffer::MAX_CONTACTS);
		PxContactPoint& contact = contactBuffer.contacts[contactCount++];
		V4StoreA(Vec4V_From_Vec3V(normal), &contact.normal.x);
		V4StoreA(Vec4V_From_Vec3V(point), &contact.point.x);
		FStore(penetration, &contact.separation);
		PX_ASSERT(contact.point.isFinite());
		PX_ASSERT(contact.normal.isFinite());
		PX_ASSERT(PxIsFinite(contact.separation));
		contact.internalFaceIndex1 = internalFaceIndex1;
	}

	PX_FORCE_INLINE bool outputSimplePCMContact(PxContactBuffer& contactBuffer, const aos::Vec3VArg point, const aos::Vec3VArg normal,
												const aos::FloatVArg penetration, PxU32 internalFaceIndex1 = PXC_CONTACT_NO_FACE_INDEX)
	{
		outputPCMContact(contactBuffer, contactBuffer.count, point, normal, penetration, internalFaceIndex1);
		return true;
	}

}//Gu
}//physx

#endif
