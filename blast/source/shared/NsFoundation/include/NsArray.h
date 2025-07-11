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
// Copyright (c) 2008-2024 NVIDIA Corporation. All rights reserved.

#ifndef NV_NSFOUNDATION_NSARRAY_H
#define NV_NSFOUNDATION_NSARRAY_H

#include "NvAssert.h"
#include "NsAllocator.h"
#include "NsBasicTemplates.h"
#include "NvIntrinsics.h"

#if NV_LINUX || NV_ANDROID || (NV_IOS && !NV_A64) || NV_OSX || NV_PS3 || NV_PSP2 || NV_WIIU
#include <tr1/type_traits>
#elif NV_WINRT || NV_XBOXONE || (NV_IOS && NV_A64) || NV_WIN64 || NV_X360 || NV_WIN32 || NV_PS4
#include <type_traits>
#if NV_IOS && NV_A64
namespace std { namespace tr1 { using std::is_pod; } }
#endif
#else
#error "OS with no defined path to type_traits.h"
#endif

#if NV_VC == 9 || NV_VC == 10
#pragma warning(push)
#pragma warning(disable : 4347) // behavior change: 'function template' is called instead of 'function'
#endif

namespace nvidia
{
namespace shdfnd
{
template <class Serializer>
void exportArray(Serializer& stream, const void* data, uint32_t size, uint32_t sizeOfElement, uint32_t capacity);
char* importArray(char* address, void** data, uint32_t size, uint32_t sizeOfElement, uint32_t capacity);

/*!
An array is a sequential container.

Implementation note
* entries between 0 and size are valid objects
* we use inheritance to build this because the array is included inline in a lot
  of objects and we want the allocator to take no space if it's not stateful, which
  aggregation doesn't allow. Also, we want the metadata at the front for the inline
  case where the allocator contains some inline storage space
*/
template <class T, class Alloc = typename AllocatorTraits<T>::Type>
class Array : protected Alloc
{
  public:
    typedef T* Iterator;
    typedef const T* ConstIterator;

    explicit Array(const NvEMPTY v) : Alloc(v)
    {
        if(mData)
            mCapacity |= NV_SIGN_BITMASK;
    }

    /*!
    Default array constructor. Initialize an empty array
    */
    NV_INLINE explicit Array(const Alloc& alloc = Alloc()) : Alloc(alloc), mData(0), mSize(0), mCapacity(0)
    {
    }

    /*!
    Initialize array with given capacity
    */
    NV_INLINE explicit Array(uint32_t size, const T& a = T(), const Alloc& alloc = Alloc())
    : Alloc(alloc), mData(0), mSize(0), mCapacity(0)
    {
        resize(size, a);
    }

    /*!
    Copy-constructor. Copy all entries from other array
    */
    template <class A>
    NV_INLINE explicit Array(const Array<T, A>& other, const Alloc& alloc = Alloc())
    : Alloc(alloc)
    {
        copy(other);
    }

    // This is necessary else the basic default copy constructor is used in the case of both arrays being of the same
    // template instance
    // The C++ standard clearly states that a template constructor is never a copy constructor [2]. In other words,
    // the presence of a template constructor does not suppress the implicit declaration of the copy constructor.
    // Also never make a copy constructor explicit, or copy-initialization* will no longer work. This is because
    // 'binding an rvalue to a const reference requires an accessible copy constructor' (http://gcc.gnu.org/bugs/)
    // *http://stackoverflow.com/questions/1051379/is-there-a-difference-in-c-between-copy-initialization-and-assignment-initializ
    NV_INLINE Array(const Array& other, const Alloc& alloc = Alloc()) : Alloc(alloc)
    {
        copy(other);
    }

    /*!
    Initialize array with given length
    */
    NV_INLINE explicit Array(const T* first, const T* last, const Alloc& alloc = Alloc())
    : Alloc(alloc), mSize(last < first ? 0 : uint32_t(last - first)), mCapacity(mSize)
    {
        mData = allocate(mSize);
        copy(mData, mData + mSize, first);
    }

    /*!
    Destructor
    */
    NV_INLINE ~Array()
    {
        destroy(mData, mData + mSize);

        if(capacity() && !isInUserMemory())
            deallocate(mData);
    }

    /*!
    Assignment operator. Copy content (deep-copy)
    */
    template <class A>
    NV_INLINE Array& operator=(const Array<T, A>& rhs)
    {
        if(&rhs == this)
            return *this;

        clear();
        reserve(rhs.mSize);
        copy(mData, mData + rhs.mSize, rhs.mData);

        mSize = rhs.mSize;
        return *this;
    }

    NV_INLINE Array& operator=(const Array& t) // Needs to be declared, see comment at copy-constructor
    {
        return operator=<Alloc>(t);
    }

    NV_FORCE_INLINE static bool isArrayOfPOD() 
    {
#if NV_VC>=14
        return std::is_trivially_copyable<T>::value;
#else
        return std::tr1::is_pod<T>::value;
#endif
    }

    /*!
    Array indexing operator.
    \param i
    The index of the element that will be returned.
    \return
    The element i in the array.
    */
    NV_FORCE_INLINE const T& operator[](uint32_t i) const
    {
        NV_ASSERT(i < mSize);
        return mData[i];
    }

    /*!
    Array indexing operator.
    \param i
    The index of the element that will be returned.
    \return
    The element i in the array.
    */
    NV_FORCE_INLINE T& operator[](uint32_t i)
    {
        NV_ASSERT(i < mSize);
        return mData[i];
    }

    /*!
    Returns a pointer to the initial element of the array.
    \return
    a pointer to the initial element of the array.
    */
    NV_FORCE_INLINE ConstIterator begin() const
    {
        return mData;
    }

    NV_FORCE_INLINE Iterator begin()
    {
        return mData;
    }

    /*!
    Returns an iterator beyond the last element of the array. Do not dereference.
    \return
    a pointer to the element beyond the last element of the array.
    */

    NV_FORCE_INLINE ConstIterator end() const
    {
        return mData + mSize;
    }

    NV_FORCE_INLINE Iterator end()
    {
        return mData + mSize;
    }

    /*!
    Returns a reference to the first element of the array. Undefined if the array is empty.
    \return a reference to the first element of the array
    */

    NV_FORCE_INLINE const T& front() const
    {
        NV_ASSERT(mSize);
        return mData[0];
    }

    NV_FORCE_INLINE T& front()
    {
        NV_ASSERT(mSize);
        return mData[0];
    }

    /*!
    Returns a reference to the last element of the array. Undefined if the array is empty
    \return a reference to the last element of the array
    */

    NV_FORCE_INLINE const T& back() const
    {
        NV_ASSERT(mSize);
        return mData[mSize - 1];
    }

    NV_FORCE_INLINE T& back()
    {
        NV_ASSERT(mSize);
        return mData[mSize - 1];
    }

    /*!
    Returns the number of entries in the array. This can, and probably will,
    differ from the array capacity.
    \return
    The number of of entries in the array.
    */
    NV_FORCE_INLINE uint32_t size() const
    {
        return mSize;
    }

    /*!
    Clears the array.
    */
    NV_INLINE void clear()
    {
        destroy(mData, mData + mSize);
        mSize = 0;
    }

    /*!
    Returns whether the array is empty (i.e. whether its size is 0).
    \return
    true if the array is empty
    */
    NV_FORCE_INLINE bool empty() const
    {
        return mSize == 0;
    }

    /*!
    Finds the first occurrence of an element in the array.
    \param a
    The element to find.
    */

    NV_INLINE Iterator find(const T& a)
    {
        uint32_t index;
        for(index = 0; index < mSize && mData[index] != a; index++)
            ;
        return mData + index;
    }

    NV_INLINE ConstIterator find(const T& a) const
    {
        uint32_t index;
        for(index = 0; index < mSize && mData[index] != a; index++)
            ;
        return mData + index;
    }

    /////////////////////////////////////////////////////////////////////////
    /*!
    Adds one element to the end of the array. Operation is O(1).
    \param a
    The element that will be added to this array.
    */
    /////////////////////////////////////////////////////////////////////////

    NV_FORCE_INLINE T& pushBack(const T& a)
    {
        if(capacity() <= mSize)
            return growAndPushBack(a);

        NV_PLACEMENT_NEW(reinterpret_cast<void*>(mData + mSize), T)(a);

        return mData[mSize++];
    }

    /////////////////////////////////////////////////////////////////////////
    /*!
    Returns the element at the end of the array. Only legal if the array is non-empty.
    */
    /////////////////////////////////////////////////////////////////////////
    NV_INLINE T popBack()
    {
        NV_ASSERT(mSize);
        T t = mData[mSize - 1];
        
        if (!isArrayOfPOD())
        {
            mData[--mSize].~T();
        } 
        else 
        {
            --mSize;        
        }

        return t;
    }

    /////////////////////////////////////////////////////////////////////////
    /*!
    Construct one element at the end of the array. Operation is O(1).
    */
    /////////////////////////////////////////////////////////////////////////
    NV_INLINE T& insert()
    {
        if(capacity() <= mSize)
            grow(capacityIncrement());

        T* ptr = mData + mSize++;
        new (ptr) T; // not 'T()' because PODs should not get default-initialized.
        return *ptr;
    }

    /////////////////////////////////////////////////////////////////////////
    /*!
    Subtracts the element on position i from the array and replace it with
    the last element.
    Operation is O(1)
    \param i
    The position of the element that will be subtracted from this array.
    */
    /////////////////////////////////////////////////////////////////////////
    NV_INLINE void replaceWithLast(uint32_t i)
    {
        NV_ASSERT(i < mSize);
        mData[i] = mData[--mSize];

        if (!isArrayOfPOD())
        {
            mData[mSize].~T();
        }
    }

    NV_INLINE void replaceWithLast(Iterator i)
    {
        replaceWithLast(static_cast<uint32_t>(i - mData));
    }

    /////////////////////////////////////////////////////////////////////////
    /*!
    Replaces the first occurrence of the element a with the last element
    Operation is O(n)
    \param a
    The position of the element that will be subtracted from this array.
    \return true if the element has been removed.
    */
    /////////////////////////////////////////////////////////////////////////

    NV_INLINE bool findAndReplaceWithLast(const T& a)
    {
        uint32_t index = 0;
        while(index < mSize && mData[index] != a)
            ++index;
        if(index == mSize)
            return false;
        replaceWithLast(index);
        return true;
    }

    /////////////////////////////////////////////////////////////////////////
    /*!
    Subtracts the element on position i from the array. Shift the entire
    array one step.
    Operation is O(n)
    \param i
    The position of the element that will be subtracted from this array.
    */
    /////////////////////////////////////////////////////////////////////////
    NV_INLINE void remove(uint32_t i)
    {
        NV_ASSERT(i < mSize);
        
        if (isArrayOfPOD())
        {
            if (i + 1 != mSize)
            {
                nvidia::intrinsics::memMove(mData + i, mData + i + 1, (mSize - i - 1) * sizeof(T));
            }
        }
        else
        {
            for(T* it = mData + i; it->~T(), ++i < mSize; ++it)
                new (it) T(mData[i]);
        }

        --mSize;
    }

    /////////////////////////////////////////////////////////////////////////
    /*!
    Removes a range from the array.  Shifts the array so order is maintained.
    Operation is O(n)
    \param begin
    The starting position of the element that will be subtracted from this array.
    \param count
    The number of elments that will be subtracted from this array.
    */
    /////////////////////////////////////////////////////////////////////////
    NV_INLINE void removeRange(uint32_t begin, uint32_t count)
    {
        NV_ASSERT(begin < mSize);
        NV_ASSERT((begin + count) <= mSize);
        
        if (!isArrayOfPOD())
        {
            for(uint32_t i = 0; i < count; i++)
            {
                mData[begin + i].~T(); // call the destructor on the ones being removed first.
            }
        }

        T* dest = &mData[begin];                       // location we are copying the tail end objects to
        T* src = &mData[begin + count];                // start of tail objects
        uint32_t move_count = mSize - (begin + count); // compute remainder that needs to be copied down

        if (isArrayOfPOD())
        {
            nvidia::intrinsics::memMove(dest, src, move_count * sizeof(T));
        }
        else
        {
            for(uint32_t i = 0; i < move_count; i++)
            {
                new (dest) T(*src); // copy the old one to the new location
                src->~T();          // call the destructor on the old location
                dest++;
                src++;
            }
        }
        mSize -= count;
    }

    //////////////////////////////////////////////////////////////////////////
    /*!
    Resize array
    */
    //////////////////////////////////////////////////////////////////////////
    NV_NOINLINE void resize(const uint32_t size, const T& a = T());

    NV_NOINLINE void resizeUninitialized(const uint32_t size);

    //////////////////////////////////////////////////////////////////////////
    /*!
    Resize array such that only as much memory is allocated to hold the
    existing elements
    */
    //////////////////////////////////////////////////////////////////////////
    NV_INLINE void shrink()
    {
        recreate(mSize);
    }

    //////////////////////////////////////////////////////////////////////////
    /*!
    Deletes all array elements and frees memory.
    */
    //////////////////////////////////////////////////////////////////////////
    NV_INLINE void reset()
    {
        resize(0);
        shrink();
    }

    //////////////////////////////////////////////////////////////////////////
    /*!
    Ensure that the array has at least size capacity.
    */
    //////////////////////////////////////////////////////////////////////////
    NV_INLINE void reserve(const uint32_t capacity)
    {
        if(capacity > this->capacity())
            grow(capacity);
    }

    //////////////////////////////////////////////////////////////////////////
    /*!
    Query the capacity(allocated mem) for the array.
    */
    //////////////////////////////////////////////////////////////////////////
    NV_FORCE_INLINE uint32_t capacity() const
    {
        return mCapacity & ~NV_SIGN_BITMASK;
    }

    //////////////////////////////////////////////////////////////////////////
    /*!
    Unsafe function to force the size of the array
    */
    //////////////////////////////////////////////////////////////////////////
    NV_FORCE_INLINE void forceSize_Unsafe(uint32_t size)
    {
        NV_ASSERT(size <= mCapacity);
        mSize = size;
    }

    //////////////////////////////////////////////////////////////////////////
    /*!
    Swap contents of an array without allocating temporary storage
    */
    //////////////////////////////////////////////////////////////////////////
    NV_INLINE void swap(Array<T, Alloc>& other)
    {
        shdfnd::swap(mData, other.mData);
        shdfnd::swap(mSize, other.mSize);
        shdfnd::swap(mCapacity, other.mCapacity);
    }

    //////////////////////////////////////////////////////////////////////////
    /*!
    Assign a range of values to this vector (resizes to length of range)
    */
    //////////////////////////////////////////////////////////////////////////
    NV_INLINE void assign(const T* first, const T* last)
    {
        resizeUninitialized(uint32_t(last - first));
        copy(begin(), end(), first);
    }

    // We need one bit to mark arrays that have been deserialized from a user-provided memory block.
    // For alignment & memory saving purpose we store that bit in the rarely used capacity member.
    NV_FORCE_INLINE uint32_t isInUserMemory() const
    {
        return mCapacity & NV_SIGN_BITMASK;
    }

    /// return reference to allocator
    NV_INLINE Alloc& getAllocator()
    {
        return *this;
    }

  protected:
    // constructor for where we don't own the memory
    Array(T* memory, uint32_t size, uint32_t capacity, const Alloc& alloc = Alloc())
    : Alloc(alloc), mData(memory), mSize(size), mCapacity(capacity | NV_SIGN_BITMASK)
    {
    }

    template <class A>
    NV_NOINLINE void copy(const Array<T, A>& other);

    NV_INLINE T* allocate(uint32_t size)
    {
        if(size > 0)
        {
            T* p = reinterpret_cast<T*>(Alloc::allocate(sizeof(T) * size, __FILE__, __LINE__));
/**
Mark a specified amount of memory with 0xcd pattern. This is used to check that the meta data
definition for serialized classes is complete in checked builds.
*/
#if NV_CHECKED
            if(p)
            {
                for(uint32_t i = 0; i < (sizeof(T) * size); ++i)
                    reinterpret_cast<uint8_t*>(p)[i] = 0xcd;
            }
#endif
            return p;
        }
        return 0;
    }

    NV_INLINE void deallocate(void* mem)
    {
        Alloc::deallocate(mem);
    }

    static NV_INLINE bool isZeroInit(const T& object)
    {
        char ZeroBuffOnStack[sizeof(object)] = {};
        return memcmp(&object, ZeroBuffOnStack, sizeof(object)) == 0;
    }

    static NV_INLINE void create(T* first, T* last, const T& a)
    {
        if (isArrayOfPOD() && isZeroInit(a))
        {
            if(last>first)
                nvidia::intrinsics::memZero(first, uint32_t((last-first) * sizeof(T)));
        }
        else
        {
            for(; first<last; ++first)
                ::new(first)T(a);
        }
    }

    static NV_INLINE void copy(T* first, T* last, const T* src)
    {           
        if (last <= first)
            return;

        if (isArrayOfPOD())
        {
            nvidia::intrinsics::memCopy(first, src, uint32_t((last-first) * sizeof(T)));
        }
        else
        {
            for(; first<last; ++first, ++src)
                ::new (first)T(*src);
        }
    }

    static NV_INLINE void destroy(T* first, T* last)
    {
        if (!isArrayOfPOD())
        {
            for(; first < last; ++first)
                first->~T();
        }
    }

    /*!
    Called when pushBack() needs to grow the array.
    \param a The element that will be added to this array.
    */
    NV_NOINLINE T& growAndPushBack(const T& a);

    /*!
    Resizes the available memory for the array.

    \param capacity
    The number of entries that the set should be able to hold.
    */
    NV_INLINE void grow(uint32_t capacity)
    {
        NV_ASSERT(this->capacity() < capacity);
        recreate(capacity);
    }

    /*!
    Creates a new memory block, copies all entries to the new block and destroys old entries.

    \param capacity
    The number of entries that the set should be able to hold.
    */
    NV_NOINLINE void recreate(uint32_t capacity);

    // The idea here is to prevent accidental bugs with pushBack or insert. Unfortunately
    // it interacts badly with InlineArrays with smaller inline allocations.
    // TODO(dsequeira): policy template arg, this is exactly what they're for.
    NV_INLINE uint32_t capacityIncrement() const
    {
        const uint32_t capacity = this->capacity();
        return capacity == 0 ? 1 : capacity * 2;
    }

    T* mData;
    uint32_t mSize;
    uint32_t mCapacity;
};

template <class T, class Alloc>
NV_NOINLINE void Array<T, Alloc>::resize(const uint32_t size, const T& a)
{
    reserve(size);
    create(mData + mSize, mData + size, a);
    destroy(mData + size, mData + mSize);
    mSize = size;
}

template <class T, class Alloc>
template <class A>
NV_NOINLINE void Array<T, Alloc>::copy(const Array<T, A>& other)
{
    if(!other.empty())
    {
        mData = allocate(mSize = mCapacity = other.size());
        copy(mData, mData + mSize, other.begin());
    }
    else
    {
        mData = NULL;
        mSize = 0;
        mCapacity = 0;
    }

    // mData = allocate(other.mSize);
    // mSize = other.mSize;
    // mCapacity = other.mSize;
    // copy(mData, mData + mSize, other.mData);
}

template <class T, class Alloc>
NV_NOINLINE void Array<T, Alloc>::resizeUninitialized(const uint32_t size)
{
    reserve(size);
    mSize = size;
}

template <class T, class Alloc>
NV_NOINLINE T& Array<T, Alloc>::growAndPushBack(const T& a)
{
    uint32_t capacity = capacityIncrement();

    T* newData = allocate(capacity);
    NV_ASSERT((!capacity) || (newData && (newData != mData)));
    copy(newData, newData + mSize, mData);

    // inserting element before destroying old array
    // avoids referencing destroyed object when duplicating array element.
    NV_PLACEMENT_NEW(reinterpret_cast<void*>(newData + mSize), T)(a);

    destroy(mData, mData + mSize);
    if(!isInUserMemory())
        deallocate(mData);

    mData = newData;
    mCapacity = capacity;

    return mData[mSize++];
}

template <class T, class Alloc>
NV_NOINLINE void Array<T, Alloc>::recreate(uint32_t capacity)
{
    T* newData = allocate(capacity);
    NV_ASSERT((!capacity) || (newData && (newData != mData)));

    copy(newData, newData + mSize, mData);
    destroy(mData, mData + mSize);
    if(!isInUserMemory())
        deallocate(mData);

    mData = newData;
    mCapacity = capacity;
}

template <class T, class Alloc>
NV_INLINE void swap(Array<T, Alloc>& x, Array<T, Alloc>& y)
{
    x.swap(y);
}

} // namespace shdfnd
} // namespace nvidia

#if NV_VC == 9 || NV_VC == 10
#pragma warning(pop)
#endif

#endif // #ifndef NV_NSFOUNDATION_NSARRAY_H
