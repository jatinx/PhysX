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

#ifndef PXG_INTEGER_AABB_H
#define PXG_INTEGER_AABB_H

#include "foundation/PxSimpleTypes.h"
#include "foundation/PxBounds3.h"
#include "hip/hip_vector_types.h"

namespace physx
{

class PxgIntegerAABB
{
public:
	enum
	{
		MIN_X = 0,
		MIN_Y,
		MIN_Z,
		MAX_X,
		MAX_Y,
		MAX_Z
	};

	/*
	\brief Return the minimum along a specified axis
	\param[in] i is the axis
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxU32	getMin(PxU32 i)	const	{	return (mMinMax)[MIN_X+i];	}

	/*
	\brief Return the maximum along a specified axis
	\param[in] i is the axis
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxU32	getMax(PxU32 i)	const	{	return (mMinMax)[MAX_X+i];	}


	/*
	\brief Encode float bounds so they are stored as integer bounds
	\param[in] bounds is the bounds to be encoded 
	\note The integer values of minima are always even, while the integer values of maxima are always odd
	\note The encoding process masks off the last four bits for minima and masks on the last four bits for maxima.
	This keeps the bounds constant when its shape is subjected to small global pose perturbations.  In turn, this helps 
	reduce computational effort in the broadphase update by reducing the amount of sorting required on near-stationary 
	bodies that are aligned along one or more axis.
	\see decode
	*/
	PX_FORCE_INLINE	void encode(const PxBounds3& bounds)
	{

		const PxU32* PX_RESTRICT min = reinterpret_cast<const PxU32*>(&bounds.minimum.x);
		const PxU32* PX_RESTRICT max = reinterpret_cast<const PxU32*>(&bounds.maximum.x);
		//Avoid min=max by enforcing the rule that mins are even and maxs are odd.
		mMinMax[MIN_X] = (encodeFloatMin(min[0]) + 1) & ~1;
		mMinMax[MIN_Y] = (encodeFloatMin(min[1]) + 1) & ~1;
		mMinMax[MIN_Z] = (encodeFloatMin(min[2]) + 1) & ~1;
		mMinMax[MAX_X] = encodeFloatMax(max[0]) | 1;
		mMinMax[MAX_Y] = encodeFloatMax(max[1]) | 1;
		mMinMax[MAX_Z] = encodeFloatMax(max[2]) | 1;
	}

	/*
	\brief Decode from integer bounds to float bounds	
	\param[out] bounds is the decoded float bounds
	\note Encode followed by decode will produce a float bound larger than the original
	due to the masking in encode.
	\see encode
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE	void decode(PxBounds3& bounds)	const
	{
		PxU32* PX_RESTRICT min = reinterpret_cast<PxU32*>(&bounds.minimum.x);
		PxU32* PX_RESTRICT max = reinterpret_cast<PxU32*>(&bounds.maximum.x);
		min[0] = decodeFloat(mMinMax[MIN_X]);
		min[1] = decodeFloat(mMinMax[MIN_Y]);
		min[2] = decodeFloat(mMinMax[MIN_Z]);
		max[0] = decodeFloat(mMinMax[MAX_X]);
		max[1] = decodeFloat(mMinMax[MAX_Y]);
		max[2] = decodeFloat(mMinMax[MAX_Z]);
	}

	/*
	\brief Encode a single minimum value from integer bounds to float bounds	
	\note The encoding process masks off the last four bits for minima
	\see encode
	*/
	static PX_CUDA_CALLABLE PX_FORCE_INLINE PxU32 encodeFloatMin(PxU32 source)
	{
		return ((encodeFloat(source) >> eGRID_SNAP_VAL) - 1) << eGRID_SNAP_VAL;
	}

	/*
	\brief Encode a single maximum value from integer bounds to float bounds	
	\note The encoding process masks on the last four bits for maxima
	\see encode
	*/
	static PX_CUDA_CALLABLE PX_FORCE_INLINE PxU32 encodeFloatMax(PxU32 source)
	{
		return ((encodeFloat(source) >> eGRID_SNAP_VAL) + 1) << eGRID_SNAP_VAL;
	}

	/*
	\brief Encode a single float value with lossless encoding to integer
	*/
	PX_CUDA_CALLABLE static PX_FORCE_INLINE PxU32 encodeFloat(PxU32 newPos)
	{
		//we may need to check on -0 and 0
		//But it should make no practical difference.
		if(newPos & PX_SIGN_BITMASK) //negative?
			return ~newPos;//reverse sequence of negative numbers
		else
			return newPos | PX_SIGN_BITMASK; // flip sign
	}

	/*
	\brief Encode a single float value with lossless encoding to integer
	*/
	PX_CUDA_CALLABLE static PX_FORCE_INLINE PxU32 decodeFloat(PxU32 ir)
	{
		if(ir & PX_SIGN_BITMASK) //positive?
			return ir & ~PX_SIGN_BITMASK; //flip sign
		else
			return ~ir; //undo reversal
	}

	/*
	\brief Shift the encoded bounds by a specified vector
	\param[in] shift is the vector used to shift the bounds
	*/
	PX_FORCE_INLINE void shift(const PxVec3& shift)
	{
		::physx::PxBounds3 elemBounds;
		decode(elemBounds);
		elemBounds.minimum -= shift;
		elemBounds.maximum -= shift;
		encode(elemBounds);
	}

	/*
	\brief Test if this aabb and another intersect
	\param[in] b is the other box
	\return True if this aabb and b intersect
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE bool intersects(const PxgIntegerAABB& b) const
	{
		return !(b.mMinMax[MIN_X] > mMinMax[MAX_X] || mMinMax[MIN_X] > b.mMinMax[MAX_X] ||
			b.mMinMax[MIN_Y] > mMinMax[MAX_Y] || mMinMax[MIN_Y] > b.mMinMax[MAX_Y] ||
			b.mMinMax[MIN_Z] > mMinMax[MAX_Z] || mMinMax[MIN_Z] > b.mMinMax[MAX_Z]);
	}

	PX_CUDA_CALLABLE PX_FORCE_INLINE bool intersects1D(const PxgIntegerAABB& b, const PxU32 axis) const
	{
		const PxU32 maxAxis = axis + 3;
		return !(b.mMinMax[axis] > mMinMax[maxAxis] || mMinMax[axis] > b.mMinMax[maxAxis]);
	}

	/*
	\brief Set the bounds to (max, max, max), (min, min, min)
	*/
	PX_CUDA_CALLABLE PX_INLINE void setEmpty()
	{
		mMinMax[MIN_X] = mMinMax[MIN_Y] = mMinMax[MIN_Z] = 0xff7fffff;  //PX_IR(PX_MAX_F32);
		mMinMax[MAX_X] = mMinMax[MAX_Y] = mMinMax[MAX_Z] = 0x00800000;	///PX_IR(0.0f);
	}

	PX_CUDA_CALLABLE PX_INLINE bool isEmpty() const
	{
		return mMinMax[MIN_X] ==  0xff7fffff && mMinMax[MIN_Y] ==  0xff7fffff && mMinMax[MIN_Z] ==  0xff7fffff && //PX_IR(PX_MAX_F32);
		mMinMax[MAX_X] ==  0x00800000 && mMinMax[MAX_Y] ==  0x00800000 &&  mMinMax[MAX_Z] == 0x00800000;	///PX_IR(0.0f);
	}

	PxU32 mMinMax[6];

private:

	enum
	{
		eGRID_SNAP_VAL = 4
	};
	
};

//This is used for the total overlap checks
struct PxgHandleRegion
{
	PxU32 handleIndex;
	PxU32 regionIndex;
};
struct PxgIntegerRegion
{
	uint4 minRange;
	uint4 maxRange;
};

}



#endif
