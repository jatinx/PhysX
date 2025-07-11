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

#include "PxsIslandSim.h"
#include "foundation/PxSort.h"
#include "foundation/PxUtilities.h"
#include "common/PxProfileZone.h"

using namespace physx;
using namespace IG;

IslandSim::IslandSim(const CPUExternalData& cpuData, GPUExternalData* gpuData, PxU64 contextID) :
	mNodes					("IslandSim::mNodes"),
	mActiveNodeIndex		("IslandSim::mActiveNodeIndex"),
	mHopCounts				("IslandSim::mHopCounts"),
	mFastRoute				("IslandSim::mFastRoute"),
	mIslandIds				("IslandSim::mIslandIds"),
	mIslands				("IslandSim::mIslands"),
	mIslandStaticTouchCount	("IslandSim.activeStaticTouchCount"),
	mActiveKinematicNodes	("IslandSim::mActiveKinematicNodes"),
	mActiveIslands			("IslandSim::mActiveIslands"),
#if IG_LIMIT_DIRTY_NODES
	mLastMapIndex			(0),
#endif
	mActivatingNodes		("IslandSim::mActivatingNodes"),
	mDestroyedEdges			("IslandSim::mDestroyedEdges"),
	mVisitedNodes			("IslandSim::mVisitedNodes"),
	mCpuData				(cpuData),
	mGpuData				(gpuData),
	mContextId				(contextID)
{
	for (PxU32 i = 0; i < Edge::eEDGE_TYPE_COUNT; ++i)
	{
		mInitialActiveNodeCount[i] = 0;
		mActiveEdgeCount[i] = 0;
	}
}

#if PX_ENABLE_ASSERTS
template <typename Thing>
static bool contains(PxArray<Thing>& arr, const Thing& thing)
{
	for(PxU32 a = 0; a < arr.size(); ++a)
	{
		if(thing == arr[a])
			return true;
	}
	return false;
}
#endif

void IslandSim::addNode(bool isActive, bool isKinematic, Node::NodeType type, PxNodeIndex nodeIndex, void* object)
{
	// PT: the nodeIndex is assigned by the SimpleIslandManager one level higher.
	const PxU32 handle = nodeIndex.index();
	{
		if(handle == mNodes.capacity())
		{
			const PxU32 newCapacity = PxMax(2*mNodes.capacity(), 256u);
			mNodes.reserve(newCapacity);
			mIslandIds.reserve(newCapacity);
			mFastRoute.reserve(newCapacity);
			mHopCounts.reserve(newCapacity);
			mActiveNodeIndex.reserve(newCapacity);
		}

		const PxU32 newSize = PxMax(handle+1, mNodes.size());
		mNodes.resize(newSize);
		mIslandIds.resize(newSize);
		mFastRoute.resize(newSize);
		mHopCounts.resize(newSize);
		mActiveNodeIndex.resize(newSize);
	}

	mActiveNodeIndex[handle] = PX_INVALID_NODE;

	Node& node = mNodes[handle];
	node.mType = PxTo8(type);
	//Ensure that the node is not currently being used.
	PX_ASSERT(node.isDeleted());

	PxU8 flags = PxU8(isActive ? 0 : Node::eREADY_FOR_SLEEPING);
	if(isKinematic)
		flags |= Node::eKINEMATIC;
	node.mFlags = flags;
	mIslandIds[handle] = IG_INVALID_ISLAND;
	mFastRoute[handle].setIndices(PX_INVALID_NODE);
	mHopCounts[handle] = 0;

	if(!isKinematic)
	{
		const IslandId islandHandle = mIslandHandles.getHandle();
		
		if(islandHandle == mIslands.capacity())
		{
			const PxU32 newCapacity = PxMax(2*mIslands.capacity(), 256u);
			mIslands.reserve(newCapacity);
			mIslandAwake.resize(newCapacity);
			mIslandStaticTouchCount.reserve(newCapacity);
		}
		const PxU32 newSize = PxMax(islandHandle+1, mIslands.size());
		mIslands.resize(newSize);
		mIslandStaticTouchCount.resize(newSize);
		mIslandAwake.growAndReset(newSize);

		Island& island = mIslands[islandHandle];
		island.mLastNode = island.mRootNode = nodeIndex;
		island.mNodeCount[type] = 1;
		mIslandIds[handle] = islandHandle;
		mIslandStaticTouchCount[islandHandle] = 0;
	}

	if(isActive)
		activateNode(nodeIndex);

	node.mObject = object;
}

// PT: preallocateConnections() and addConnectionPreallocated() are used to replicate IslandSim::addConnection() multi-threaded
void IslandSim::preallocateConnections(EdgeIndex handle)
{
	if(handle >= mEdges.capacity())
	{
		PX_PROFILE_ZONE("ReserveIslandEdges", mContextId);
		const PxU32 newSize = handle + 2048;
		mEdges.reserve(newSize);
		if(mGpuData)
			mGpuData->mActiveContactEdges.resize(mEdges.capacity());
	}
	mEdges.resize(PxMax(mEdges.size(), handle+1));
	if(mGpuData)
		mGpuData->mActiveContactEdges.reset(handle);
}

bool IslandSim::addConnectionPreallocated(PxNodeIndex nodeHandle1, PxNodeIndex nodeHandle2, Edge::EdgeType edgeType, EdgeIndex handle)
{
	// PT: the EdgeIndex is assigned by the SimpleIslandManager one level higher.

	PX_UNUSED(nodeHandle1);
	PX_UNUSED(nodeHandle2);
	Edge& edge = mEdges[handle];

	if(edge.isPendingDestroyed())
	{
		//If it's in this state, then the edge has been tagged for destruction but actually is now not needed to be destroyed
		edge.clearPendingDestroyed();
		return false;
	}

	if(edge.isInDirtyList())
	{
		PX_ASSERT(mCpuData.mEdgeNodeIndices[handle * 2].index() == nodeHandle1.index());
		PX_ASSERT(mCpuData.mEdgeNodeIndices[handle * 2 + 1].index() == nodeHandle2.index());
		PX_ASSERT(edge.mEdgeType == edgeType);
		return false;
	}

	PX_ASSERT(!edge.isInserted());

	PX_ASSERT(edge.isDestroyed());
	edge.clearDestroyed();

	PX_ASSERT(edge.mNextIslandEdge == IG_INVALID_ISLAND);
	PX_ASSERT(edge.mPrevIslandEdge == IG_INVALID_ISLAND);

	PX_ASSERT(mEdgeInstances.size() <= 2*handle || mEdgeInstances[2*handle].mNextEdge == IG_INVALID_EDGE);
	PX_ASSERT(mEdgeInstances.size() <= 2*handle || mEdgeInstances[2*handle+1].mNextEdge == IG_INVALID_EDGE);
	PX_ASSERT(mEdgeInstances.size() <= 2*handle || mEdgeInstances[2*handle].mPrevEdge == IG_INVALID_EDGE);
	PX_ASSERT(mEdgeInstances.size() <= 2*handle || mEdgeInstances[2*handle+1].mPrevEdge == IG_INVALID_EDGE);

	edge.mEdgeType = PxTo16(edgeType);

	PX_ASSERT(handle*2 >= mEdgeInstances.size() || mEdgeInstances[handle*2].mNextEdge == IG_INVALID_EDGE);
	PX_ASSERT(handle*2+1 >= mEdgeInstances.size() || mEdgeInstances[handle*2+1].mNextEdge == IG_INVALID_EDGE);
	PX_ASSERT(handle*2 >= mEdgeInstances.size() || mEdgeInstances[handle*2].mPrevEdge == IG_INVALID_EDGE);
	PX_ASSERT(handle*2+1 >= mEdgeInstances.size() || mEdgeInstances[handle*2+1].mPrevEdge == IG_INVALID_EDGE);
	
	//Add the new handle
	PX_ASSERT(!edge.isInDirtyList());	// PT: otherwise it should have exited the function above
	PX_ASSERT(!contains(mDirtyEdges[edgeType], handle));
	// PT: TODO: we could push back to an array MT but that would break determinism
	//mDirtyEdges[edgeType].pushBack(handle);
	edge.markInDirtyList();

	edge.mEdgeState &= ~(Edge::eACTIVATING);
	return true;
}

void IslandSim::addConnection(PxNodeIndex nodeHandle1, PxNodeIndex nodeHandle2, Edge::EdgeType edgeType, EdgeIndex handle)
{
	// PT: the EdgeIndex is assigned by the SimpleIslandManager one level higher.

	preallocateConnections(handle);

	if(addConnectionPreallocated(nodeHandle1, nodeHandle2, edgeType, handle))
		mDirtyEdges[edgeType].pushBack(handle);
}

// PT: last part of IslandSim::addConnection, not MT in IslandSim::addConnectionPreallocated
void IslandSim::addDelayedDirtyEdges(PxU32 nbHandles, const EdgeIndex* handles)
{
	// PT: TODO: better version
	while(nbHandles--)
	{
		const EdgeIndex h = *handles++;
		const Edge& edge = mEdges[h];
		mDirtyEdges[edge.mEdgeType].pushBack(h);
	}
}

void IslandSim::addConnectionToGraph(EdgeIndex handle)
{
	const EdgeInstanceIndex instanceHandle = 2*handle;
	PX_ASSERT(instanceHandle < mEdgeInstances.capacity());
	/*if(instanceHandle == mEdgeInstances.capacity())
	{
		mEdgeInstances.reserve(2*mEdgeInstances.capacity() + 2);
	}*/
	mEdgeInstances.resize(PxMax(instanceHandle+2, mEdgeInstances.size()));

	Edge& edge = mEdges[handle];
	
	// PT: TODO: int bools
	PxIntBool activeEdge = false;
	bool kinematicKinematicEdge = true;

	const PxNodeIndex nodeIndex1 = mCpuData.mEdgeNodeIndices[instanceHandle];
	const PxNodeIndex nodeIndex2 = mCpuData.mEdgeNodeIndices[instanceHandle+1];

	struct Local
	{
		static PX_FORCE_INLINE void connectEdge(Cm::BlockArray<EdgeInstance>& edgeInstances, EdgeInstanceIndex edgeIndex, Node& source)
		{
			EdgeInstance& instance = edgeInstances[edgeIndex];

			PX_ASSERT(instance.mNextEdge == IG_INVALID_EDGE);
			PX_ASSERT(instance.mPrevEdge == IG_INVALID_EDGE);

			instance.mNextEdge = source.mFirstEdgeIndex;
			if(source.mFirstEdgeIndex != IG_INVALID_EDGE)
			{
				EdgeInstance& firstEdge = edgeInstances[source.mFirstEdgeIndex];
				firstEdge.mPrevEdge = edgeIndex;
			}

			source.mFirstEdgeIndex = edgeIndex;
			instance.mPrevEdge = IG_INVALID_EDGE;
		}
	};

	const PxU32 index1 = nodeIndex1.index();
	if(index1 != PX_INVALID_NODE)
	{
		Node& node = mNodes[index1];
		Local::connectEdge(mEdgeInstances, instanceHandle, node);
		activeEdge = node.isActiveOrActivating();
		kinematicKinematicEdge = node.isKinematic();
	}

	const PxU32 index2 = nodeIndex2.index();
	if(index1 != index2 && index2 != PX_INVALID_NODE)
	{
		Node& node = mNodes[index2];
		Local::connectEdge(mEdgeInstances, instanceHandle + 1, node);
		activeEdge |= node.isActiveOrActivating();
		kinematicKinematicEdge = kinematicKinematicEdge && node.isKinematic();
	}

	if(activeEdge && (!kinematicKinematicEdge || edge.getEdgeType() == IG::Edge::eCONTACT_MANAGER))
	{				
		markEdgeActive(handle, nodeIndex1, nodeIndex2);
		edge.activateEdge();
	}
}

void IslandSim::removeConnectionFromGraph(EdgeIndex edgeIndex)
{
	const PxNodeIndex nodeIndex1 = mCpuData.mEdgeNodeIndices[2 * edgeIndex];
	const PxNodeIndex nodeIndex2 = mCpuData.mEdgeNodeIndices[2 * edgeIndex + 1];

	const PxU32 index1 = nodeIndex1.index();
	const PxU32 index2 = nodeIndex2.index();

	if (index1 != PX_INVALID_NODE)
	{
		Node& node = mNodes[index1];
		if (index2 == mFastRoute[index1].index())
			mFastRoute[index1].setIndices(PX_INVALID_NODE);
		if(!node.isDirty())
		{
			//mDirtyNodes.pushBack(nodeIndex1);
			mDirtyMap.growAndSet(index1);
			node.markDirty();
		}
	}

	if (index2 != PX_INVALID_NODE)
	{
		Node& node = mNodes[index2];
		if (index1 == mFastRoute[index2].index())
			mFastRoute[index2].setIndices(PX_INVALID_NODE);
		if(!node.isDirty())
		{
			mDirtyMap.growAndSet(index2);
			node.markDirty();
		}
	}
}

void IslandSim::removeConnection(EdgeIndex edgeIndex)
{
	Edge& edge = mEdges[edgeIndex];
	if(!edge.isPendingDestroyed())// && edge.isInserted())
	{
		mDestroyedEdges.pushBack(edgeIndex);
		/*if(!edge.isInserted())
			edge.setReportOnlyDestroy();*/
	}
	edge.setPendingDestroyed();
}

void IslandSim::removeConnectionInternal(EdgeIndex edgeIndex)
{
	PX_ASSERT(edgeIndex != IG_INVALID_EDGE);
	const EdgeInstanceIndex edgeInstanceBase = edgeIndex*2;
	const PxNodeIndex nodeIndex1 = mCpuData.mEdgeNodeIndices[edgeIndex * 2];
	const PxNodeIndex nodeIndex2 = mCpuData.mEdgeNodeIndices[edgeIndex * 2 + 1];

	struct Local
	{
		static void disconnectEdge(Cm::BlockArray<EdgeInstance>& edgeInstances, EdgeInstanceIndex edgeIndex, Node& node)
		{
			EdgeInstance& instance = edgeInstances[edgeIndex];

			PX_ASSERT(instance.mNextEdge == IG_INVALID_EDGE || edgeInstances[instance.mNextEdge].mPrevEdge == edgeIndex);
			PX_ASSERT(instance.mPrevEdge == IG_INVALID_EDGE || edgeInstances[instance.mPrevEdge].mNextEdge == edgeIndex);

			if(node.mFirstEdgeIndex == edgeIndex)
			{
				PX_ASSERT(instance.mPrevEdge == IG_INVALID_EDGE);
				node.mFirstEdgeIndex = instance.mNextEdge;
			}
			else
			{
				EdgeInstance& prev = edgeInstances[instance.mPrevEdge];
				PX_ASSERT(prev.mNextEdge == edgeIndex);
				prev.mNextEdge = instance.mNextEdge;
			}

			if(instance.mNextEdge != IG_INVALID_EDGE)
			{
				EdgeInstance& next = edgeInstances[instance.mNextEdge];
				PX_ASSERT(next.mPrevEdge == edgeIndex);
				next.mPrevEdge = instance.mPrevEdge;
			}

			PX_ASSERT(instance.mNextEdge == IG_INVALID_EDGE || edgeInstances[instance.mNextEdge].mPrevEdge == instance.mPrevEdge);
			PX_ASSERT(instance.mPrevEdge == IG_INVALID_EDGE || edgeInstances[instance.mPrevEdge].mNextEdge == instance.mNextEdge);

			instance.mNextEdge = IG_INVALID_EDGE;
			instance.mPrevEdge = IG_INVALID_EDGE;
		}
	};

	const PxU32 index1 = nodeIndex1.index();
	const PxU32 index2 = nodeIndex2.index();

	if (index1 != PX_INVALID_NODE)
		Local::disconnectEdge(mEdgeInstances, edgeInstanceBase, mNodes[index1]);

	if (index2 != PX_INVALID_NODE && index1 != index2)
		Local::disconnectEdge(mEdgeInstances, edgeInstanceBase + 1, mNodes[index2]);
}

void IslandSim::activateNode(PxNodeIndex nodeIndex)
{
	const PxU32 index = nodeIndex.index();
	if(index != PX_INVALID_NODE)
	{
		Node& node = mNodes[index];

		if(!node.isActiveOrActivating())
		{
			//If the node is kinematic and already in the active node list, then we need to remove it
			//from the active kinematic node list, then re-add it after the wake-up. It's a bit dumb
			//but it means that we don't need another index

			if(node.isKinematic() && mActiveNodeIndex[index] != PX_INVALID_NODE)
			{
				//node.setActive();
				//node.clearIsReadyForSleeping(); //Clear the "isReadyForSleeping" flag. Just in case it was set
				//return;

				const PxU32 activeRefCount = node.mActiveRefCount;
				node.mActiveRefCount = 0;
				node.clearActive();
				markKinematicInactive(nodeIndex);
				node.mActiveRefCount = activeRefCount;
			}
			
			node.setActivating(); //Tag it as activating
			PX_ASSERT(mActiveNodeIndex[index] == PX_INVALID_NODE);
			mActiveNodeIndex[index] = mActivatingNodes.size();	
			//Add to waking list
			mActivatingNodes.pushBack(nodeIndex);
		}
		node.clearIsReadyForSleeping(); //Clear the "isReadyForSleeping" flag. Just in case it was set
	}
}

void IslandSim::deactivateNode(PxNodeIndex nodeIndex)
{
	const PxU32 index = nodeIndex.index();
	if(index != PX_INVALID_NODE)
	{
		Node& node = mNodes[index];

		//If the node is activating, clear its activating state and remove it from the activating list. 
		//If it wasn't already activating, then it's probably already in the active list

		const PxIntBool wasActivating = node.isActivating();

		if(wasActivating)
		{
			//Already activating, so remove it from the activating list
			node.clearActivating();
			PX_ASSERT(mActivatingNodes[mActiveNodeIndex[index]].index() == index);
			const PxNodeIndex replaceIndex = mActivatingNodes[mActivatingNodes.size()-1];
			mActiveNodeIndex[replaceIndex.index()] = mActiveNodeIndex[index];
			mActivatingNodes[mActiveNodeIndex[index]] = replaceIndex;
			mActivatingNodes.forceSize_Unsafe(mActivatingNodes.size()-1);
			mActiveNodeIndex[index] = PX_INVALID_NODE;

			if(node.isKinematic())
			{
				//If we were temporarily removed from the active kinematic list to be put in the waking kinematic list
				//then add the node back in before deactivating the node. This is a bit counter-intuitive but the active
				//kinematic list contains all active kinematics and all kinematics that are referenced by an active constraint
				PX_ASSERT(mActiveNodeIndex[index] == PX_INVALID_NODE);
				mActiveNodeIndex[index] = mActiveKinematicNodes.size();
				mActiveKinematicNodes.pushBack(nodeIndex);
			}
		}

		//Raise the "ready for sleeping" flag so that island gen can put this node to sleep
		node.setIsReadyForSleeping();
	}
}

void IslandSim::putNodeToSleep(PxNodeIndex nodeIndex)
{
	if(nodeIndex.index() != PX_INVALID_NODE)
		deactivateNode(nodeIndex);
}

PX_FORCE_INLINE void IslandSim::makeEdgeActive(EdgeInstanceIndex index, bool testEdgeType)
{
	const EdgeIndex idx = index / 2;
	Edge& edge = mEdges[idx];
	if (!edge.isActive() && (!testEdgeType || (edge.getEdgeType() != IG::Edge::eCONSTRAINT)))
	{
		//Make the edge active...
		const PxNodeIndex nodeIndex1 = mCpuData.mEdgeNodeIndices[idx * 2];
		const PxNodeIndex nodeIndex2 = mCpuData.mEdgeNodeIndices[idx * 2 + 1];

		PX_ASSERT(nodeIndex1.index() == PX_INVALID_NODE || !mNodes[nodeIndex1.index()].isActive() || mNodes[nodeIndex1.index()].isKinematic());
		PX_ASSERT(nodeIndex2.index() == PX_INVALID_NODE || !mNodes[nodeIndex2.index()].isActive() || mNodes[nodeIndex2.index()].isKinematic());

		markEdgeActive(idx, nodeIndex1, nodeIndex2);
		edge.activateEdge();
	}
}

void IslandSim::activateNodeInternal(PxNodeIndex nodeIndex)
{
	//This method should activate the node, then activate all the connections involving this node
	Node& node = mNodes[nodeIndex.index()];

	if(!node.isActive())
	{
		PX_ASSERT(mActiveNodeIndex[nodeIndex.index()] == PX_INVALID_NODE);

		//Activate all the edges + nodes...

		EdgeInstanceIndex index = node.mFirstEdgeIndex;

		while(index != IG_INVALID_EDGE)
		{
			makeEdgeActive(index, false);

			index = mEdgeInstances[index].mNextEdge;
		}

		if(node.isKinematic())
			markKinematicActive(nodeIndex);
		else
			markActive(nodeIndex);

		node.setActive();
	}
}

void IslandSim::deactivateNodeInternal(PxNodeIndex nodeIndex)
{
	//We deactivate a node, we need to loop through all the edges and deactivate them *if* both bodies are asleep

	Node& node = mNodes[nodeIndex.index()];

	if(node.isActive())
	{
		if(node.isKinematic())
			markKinematicInactive(nodeIndex);
		else
			markInactive(nodeIndex);

		//Clear the active status flag
		node.clearActive();
		node.clearActivating();

		EdgeInstanceIndex index = node.mFirstEdgeIndex;

		while(index != IG_INVALID_EDGE)
		{
			const EdgeInstance& instance = mEdgeInstances[index];

			const PxNodeIndex outboundNode = mCpuData.mEdgeNodeIndices[index ^ 1];
			if(outboundNode.index() == PX_INVALID_NODE || 
				!mNodes[outboundNode.index()].isActive())
			{
				const EdgeIndex idx = index/2;
				Edge& edge = mEdges[idx]; //InstanceIndex/2 = edgeIndex
				//PX_ASSERT(edge.isActive()); //The edge must currently be inactive because the node was active
				//Deactivate the edge if both nodes connected are inactive OR if one node is static/kinematic and the other is inactive...
				PX_ASSERT(mCpuData.mEdgeNodeIndices[index & (~1)].index() == PX_INVALID_NODE || !mNodes[mCpuData.mEdgeNodeIndices[index & (~1)].index()].isActive());
				PX_ASSERT(mCpuData.mEdgeNodeIndices[index | 1].index() == PX_INVALID_NODE || !mNodes[mCpuData.mEdgeNodeIndices[index | 1].index()].isActive());
				if(edge.isActive())
				{
					edge.deactivateEdge();
					mActiveEdgeCount[edge.mEdgeType]--;
					removeEdgeFromActivatingList(idx);
					mDeactivatingEdges[edge.mEdgeType].pushBack(idx);
				}
			}
			index = instance.mNextEdge;
		}		
	}
}

#if IG_SANITY_CHECKS
bool IslandSim::canFindRoot(PxNodeIndex startNode, PxNodeIndex targetNode, PxArray<PxNodeIndex>* visitedNodes)
{
	if(visitedNodes)
		visitedNodes->pushBack(startNode);
	if(startNode.index() == targetNode.index())
		return true;
	PxBitMap visitedState;
	visitedState.resizeAndClear(mNodes.size());

	PxArray<PxNodeIndex> stack;

	stack.pushBack(startNode);

	visitedState.set(startNode.index());

	do
	{
		const PxNodeIndex currentIndex = stack.popBack();
		const Node& currentNode = mNodes[currentIndex.index()];

		EdgeInstanceIndex currentEdge = currentNode.mFirstEdgeIndex;

		while(currentEdge != IG_INVALID_EDGE)
		{
			const EdgeInstance& edge = mEdgeInstances[currentEdge];
			const PxNodeIndex outboundNode = mCpuData.mEdgeNodeIndices[currentEdge ^ 1];
			if(outboundNode.index() != PX_INVALID_NODE && !mNodes[outboundNode.index()].isKinematic() && !visitedState.test(outboundNode.index()))
			{
				if(outboundNode.index() == targetNode.index())
					return true;

				visitedState.set(outboundNode.index());
				stack.pushBack(outboundNode);
				if(visitedNodes)
					visitedNodes->pushBack(outboundNode);
			}

			currentEdge = edge.mNextEdge;
		}
	}
	while(stack.size());

	return false;
}
#endif

void IslandSim::unwindRoute(PxU32 traversalIndex, PxNodeIndex lastNode, PxU32 hopCount, IslandId id)
{
	//We have found either a witness *or* the root node with this traversal. In the event of finding the root node, hopCount will be 0. In the event of finding
	//a witness, hopCount will be the hopCount that witness reported as being the distance to the root.

	PxU32 currIndex = traversalIndex;
	PxU32 hc = hopCount+1; //Add on 1 for the hop to the witness/root node.
	do
	{
		TraversalState& state = mVisitedNodes[currIndex];
		mHopCounts[state.mNodeIndex.index()] = hc++;
		mIslandIds[state.mNodeIndex.index()] = id;
		mFastRoute[state.mNodeIndex.index()] = lastNode;
		currIndex = state.mPrevIndex;
		lastNode = state.mNodeIndex;
	}
	while(currIndex != PX_INVALID_NODE);
}

void IslandSim::activateIslandInternal(const Island& island)
{
	PxNodeIndex currentNode = island.mRootNode;
	while(currentNode.index() != PX_INVALID_NODE)
	{
		activateNodeInternal(currentNode);
		currentNode = mNodes[currentNode.index()].mNextNode;
	}
}

void IslandSim::activateIsland(IslandId islandId)
{
	Island& island = mIslands[islandId];
	PX_ASSERT(!mIslandAwake.test(islandId));
	PX_ASSERT(island.mActiveIndex == IG_INVALID_ISLAND);
	
	activateIslandInternal(island);

	markIslandActive(islandId);
}

void IslandSim::deactivateIsland(IslandId islandId)
{
	PX_ASSERT(mIslandAwake.test(islandId));
	Island& island = mIslands[islandId];
	
	PxNodeIndex currentNode = island.mRootNode;
	while(currentNode.index() != PX_INVALID_NODE)
	{
		const Node& node = mNodes[currentNode.index()];
		
		//if(mActiveNodeIndex[currentNode.index()] < mInitialActiveNodeCount[node.mType])
		mNodesToPutToSleep[node.mType].pushBack(currentNode); //If this node was previously active, then push it to the list of nodes to deactivate
		deactivateNodeInternal(currentNode);
		currentNode = node.mNextNode;
	}
	markIslandInactive(islandId);
}

void IslandSim::wakeIslandsInternal(bool flag)
{
	//(1) Iterate over activating nodes and activate them

	const PxU32 originalActiveIslands = mActiveIslands.size();

	if(flag)
	{
		for (PxU32 a = 0; a < Edge::eEDGE_TYPE_COUNT; ++a)
		{
			for (PxU32 i = 0, count = mActivatedEdges[a].size(); i < count; ++i)
			{
				IG::Edge& edge = mEdges[mActivatedEdges[a][i]];
				edge.mEdgeState &= (~Edge::eACTIVATING);
			}

			mActivatedEdges[a].forceSize_Unsafe(0);
		}

		for (PxU32 a = 0; a < Edge::eEDGE_TYPE_COUNT; ++a)
		{
			mInitialActiveNodeCount[a] = mActiveNodes[a].size();
		}
	}

	for(PxU32 a = 0; a < mActivatingNodes.size(); ++a)
	{
		const PxNodeIndex wakeNode = mActivatingNodes[a];

		const IslandId islandId = mIslandIds[wakeNode.index()];

		Node& node = mNodes[wakeNode.index()];
		node.clearActivating();
		if(islandId != IG_INVALID_ISLAND)
		{
			if(!mIslandAwake.test(islandId))
				markIslandActive(islandId);						

			mActiveNodeIndex[wakeNode.index()] = PX_INVALID_NODE; //Mark active node as invalid.
			activateNodeInternal(wakeNode);
		}
		else
		{
			PX_ASSERT(node.isKinematic());
			node.setActive();
			PX_ASSERT(mActiveNodeIndex[wakeNode.index()] == a);
			mActiveNodeIndex[wakeNode.index()] = mActiveKinematicNodes.size();
			mActiveKinematicNodes.pushBack(wakeNode);

			//Wake up the islands connected to this waking kinematic!
			EdgeInstanceIndex index = node.mFirstEdgeIndex;
			while(index != IG_INVALID_EDGE)
			{
				const EdgeInstance& edgeInstance = mEdgeInstances[index];

				const PxNodeIndex outboundNode = mCpuData.mEdgeNodeIndices[index ^ 1];
				//Edge& edge = mEdges[index/2];
				//if(edge.isConnected()) //Only wake up if the edge is not connected...
				const PxNodeIndex nodeIndex = outboundNode;

				if (nodeIndex.isStaticBody() || mIslandIds[nodeIndex.index()] == IG_INVALID_ISLAND)
				{
					//If the edge connects to a static body *or* it connects to a node which is not part of an island (i.e. a kinematic), then activate the edge
					makeEdgeActive(index, true);
				}
				else
				{
					const IslandId connectedIslandId = mIslandIds[nodeIndex.index()];
					if(!mIslandAwake.test(connectedIslandId))
					{
						//Wake up that island
						markIslandActive(connectedIslandId);
					}
				}

				index = edgeInstance.mNextEdge;
			}
		}
	}

	mActivatingNodes.forceSize_Unsafe(0);

	for(PxU32 a = originalActiveIslands; a < mActiveIslands.size(); ++a)
		activateIslandInternal(mIslands[mActiveIslands[a]]);
}

void IslandSim::wakeIslands()
{
	PX_PROFILE_ZONE("Basic.wakeIslands", mContextId);
	wakeIslandsInternal(true);
}

void IslandSim::wakeIslands2()
{
	PX_PROFILE_ZONE("Basic.wakeIslands2", mContextId);
	wakeIslandsInternal(false);
}

void IslandSim::insertNewEdges()
{
	PX_PROFILE_ZONE("Basic.insertNewEdges", mContextId);

	mEdgeInstances.reserve(mEdges.capacity()*2);
	
	for(PxU32 i = 0; i < Edge::eEDGE_TYPE_COUNT; ++i)
	{
		for(PxU32 a = 0; a < mDirtyEdges[i].size(); ++a)
		{
			const EdgeIndex edgeIndex = mDirtyEdges[i][a];

			Edge& edge = mEdges[edgeIndex];

			if(!edge.isPendingDestroyed())
			{
				//PX_ASSERT(!edge.isInserted());
				if(!edge.isInserted())
				{
					addConnectionToGraph(edgeIndex);
					edge.setInserted();
				}
			}
		}
	}
}

void IslandSim::removeDestroyedEdges()
{
	PX_PROFILE_ZONE("Basic.removeDestroyedEdges", mContextId);

	for(PxU32 a = 0; a < mDestroyedEdges.size(); ++a)
	{
		const EdgeIndex edgeIndex = mDestroyedEdges[a];

		const Edge& edge = mEdges[edgeIndex];
		
		if(edge.isPendingDestroyed())
		{
			if(!edge.isInDirtyList() && edge.isInserted())
			{
				removeConnectionInternal(edgeIndex);
				removeConnectionFromGraph(edgeIndex);
				//edge.clearInserted();
			}
			//edge.clearDestroyed();
		}
	}		
}

IslandId IslandSim::addNodeToIsland(PxNodeIndex nodeIndex1, PxNodeIndex nodeIndex2, IslandId islandId2, bool active1, bool active2)
{
	PX_ASSERT(islandId2 != IG_INVALID_ISLAND);
	if (nodeIndex1.index() != PX_INVALID_NODE)
	{
		if (!mNodes[nodeIndex1.index()].isKinematic())
		{
			//We need to add node 1 to island2
			PX_ASSERT(mNodes[nodeIndex1.index()].mNextNode.index() == PX_INVALID_NODE); //Ensure that this node is not in any other island
			PX_ASSERT(mNodes[nodeIndex1.index()].mPrevNode.index() == PX_INVALID_NODE); //Ensure that this node is not in any other island
							
			Island& island = mIslands[islandId2];

			Node& lastNode = mNodes[island.mLastNode.index()];

			PX_ASSERT(lastNode.mNextNode.index() == PX_INVALID_NODE);

			Node& node = mNodes[nodeIndex1.index()];
			lastNode.mNextNode = nodeIndex1;
			node.mPrevNode = island.mLastNode;
			island.mLastNode = nodeIndex1;
			island.mNodeCount[node.mType]++;
			mIslandIds[nodeIndex1.index()] = islandId2;
			mHopCounts[nodeIndex1.index()] = mHopCounts[nodeIndex2.index()] + 1;
			mFastRoute[nodeIndex1.index()] = nodeIndex2;

			if(active1 || active2)
			{
				if(!mIslandAwake.test(islandId2))
				{
					//This island wasn't already awake, so need to wake the whole island up
					activateIsland(islandId2);
				}
				if(!active1)
				{
					//Wake up this node...
					activateNodeInternal(nodeIndex1);
				}
			}
		}
		else if(active1 && !active2)
		{
			//Active kinematic object -> wake island!
			activateIsland(islandId2);
		}
	}
	else
	{
		//A new touch with a static body...
		Node& node = mNodes[nodeIndex2.index()];
		node.mStaticTouchCount++; //Increment static touch counter on the body
		//Island& island = mIslands[islandId2];
		//island.mStaticTouchCount++; //Increment static touch counter on the island
		mIslandStaticTouchCount[islandId2]++;
	}
	return islandId2;
}

void IslandSim::processNewEdges()
{
	PX_PROFILE_ZONE("Basic.processNewEdges", mContextId);
	//Stage 1: we process the list of new pairs. To do this, we need to first sort them based on a predicate...

	insertNewEdges();

	mHopCounts.resize(mNodes.size()); //Make sure we have enough space for hop counts for all nodes
	mFastRoute.resize(mNodes.size());

	for(PxU32 i = 0; i < Edge::eEDGE_TYPE_COUNT; ++i)
	{
		for(PxU32 a = 0; a < mDirtyEdges[i].size(); ++a)
		{
			const EdgeIndex edgeIndex = mDirtyEdges[i][a];
			const Edge& edge = mEdges[edgeIndex];
			
			/*PX_ASSERT(edge.mState != Edge::eDESTROYED || ((edge.mNode1.index() == PX_INVALID_NODE || mNodes[edge.mNode1.index()].isKinematic() || mNodes[edge.mNode1.index()].isActive() == false) &&
				(edge.mNode2.index() == PX_INVALID_NODE || mNodes[edge.mNode2.index()].isKinematic() || mNodes[edge.mNode2.index()].isActive() == false)));*/

			//edge.clearInDirtyList();
			
			//We do not process either destroyed or disconnected edges
			if(/*edge.isConnected() && */!edge.isPendingDestroyed())
			{
				//Conditions:
				//(1)	Neither body is in an island (static/kinematics are never in islands) so we need to create a new island containing these bodies 
				//		or just 1 body if the other is kinematic/static
				//(2)	Both bodies are already in the same island. Update root node hop count estimates for the bodies if a route through the new connection
				//		is shorter for either body
				//(3)	One body is already in an island and the other isn't, so we just add the new body to the existing island.
				//(4)	Both bodies are in different islands. In that case, we merge the islands

				const PxNodeIndex nodeIndex1 = mCpuData.mEdgeNodeIndices[2 * edgeIndex];
				const PxNodeIndex nodeIndex2 = mCpuData.mEdgeNodeIndices[2 * edgeIndex+1];

				const PxU32 index1 = nodeIndex1.index();
				const PxU32 index2 = nodeIndex2.index();

				const IslandId islandId1 = index1 == PX_INVALID_NODE ? IG_INVALID_ISLAND : mIslandIds[index1];
				const IslandId islandId2 = index2 == PX_INVALID_NODE ? IG_INVALID_ISLAND : mIslandIds[index2];

				//TODO - wake ups!!!!
				//If one of the nodes is awake and the other is asleep, we need to wake 'em up

				//When a node is activated, the island must also be activated...

				const bool active1 = index1 != PX_INVALID_NODE && mNodes[index1].isActive();
				const bool active2 = index2 != PX_INVALID_NODE && mNodes[index2].isActive();

				IslandId islandId = IG_INVALID_ISLAND;

				if(islandId1 == IG_INVALID_ISLAND && islandId2 == IG_INVALID_ISLAND)
				{
					//All nodes should be introduced in an island now unless they are static or kinematic. Therefore, if we get here, we have an edge
					//between 2 kinematic nodes or a kinematic and static node. These should not influence island management so we should just ignore
					//these edges.
				}
				else if(islandId1 == islandId2)
				{
					islandId = islandId1;
					if(active1 || active2)
					{
						PX_ASSERT(mIslandAwake.test(islandId1)); //If we got here, where the 2 were already in an island, if 1 node is awake, the whole island must be awake
					}
					//Both bodies in the same island. Nothing major to do already but we should see if this creates a shorter path to root for either node
					const PxU32 hopCount1 = mHopCounts[index1];
					const PxU32 hopCount2 = mHopCounts[index2];
					if((hopCount1+1) < hopCount2)
					{
						//It would be faster for node 2 to go through node 1
						mHopCounts[index2] = hopCount1 + 1;
						mFastRoute[index2] = nodeIndex1;
					}
					else if((hopCount2+1) < hopCount1)
					{
						//It would be faster for node 1 to go through node 2
						mHopCounts[index1] = hopCount2 + 1;
						mFastRoute[index1] = nodeIndex2;
					}

					//No need to activate/deactivate the island. Its state won't have changed
				}
				else if(islandId1 == IG_INVALID_ISLAND)
				{
					islandId = addNodeToIsland(nodeIndex1, nodeIndex2, islandId2, active1, active2);
				}
				else if (islandId2 == IG_INVALID_ISLAND)
				{
					islandId = addNodeToIsland(nodeIndex2, nodeIndex1, islandId1, active2, active1);
				}
				else
				{
					PX_ASSERT(islandId1 != islandId2);
					PX_ASSERT(islandId1 != IG_INVALID_ISLAND && islandId2 != IG_INVALID_ISLAND);

					if(active1 || active2)
					{
						//One of the 2 islands was awake, so need to wake the other one! We do this now, before we merge the islands, to ensure that all 
						//the bodies are activated
						if(!mIslandAwake.test(islandId1))
						{
							//This island wasn't already awake, so need to wake the whole island up
							activateIsland(islandId1);
						}
						if(!mIslandAwake.test(islandId2))
						{
							//This island wasn't already awake, so need to wake the whole island up
							activateIsland(islandId2);
						}
					}

					//OK. We need to merge these islands together...
					islandId = mergeIslands(islandId1, islandId2, nodeIndex1, nodeIndex2);
				}

				if(islandId != IG_INVALID_ISLAND)
				{
					//Add new edge to existing island
					Island& island = mIslands[islandId];
					addEdgeToIsland(island, edgeIndex);
				}
			}
		}
	}
}

#if PX_DEBUG
bool IslandSim::isPathTo(PxNodeIndex startNode, PxNodeIndex targetNode) const
{
	const Node& node = mNodes[startNode.index()];

	EdgeInstanceIndex index = node.mFirstEdgeIndex;
	while(index != IG_INVALID_EDGE)
	{
		const EdgeInstance& instance = mEdgeInstances[index];
		if(/*mEdges[index/2].isConnected() &&*/ mCpuData.mEdgeNodeIndices[index^1].index() == targetNode.index())
			return true;
		index = instance.mNextEdge;
	}
	return false;
}
#endif

bool IslandSim::tryFastPath(PxNodeIndex startNode, PxNodeIndex targetNode, IslandId islandId)
{
	PX_UNUSED(startNode);
	PX_UNUSED(targetNode);

	PxNodeIndex currentNode = startNode;

	const PxU32 currentVisitedNodes = mVisitedNodes.size();

	PxU32 depth = 0;
	
	bool found = false;
	do
	{
		//Get the fast path from this node...
		
		if(mVisitedState.test(currentNode.index()))
		{
			found = mIslandIds[currentNode.index()] != IG_INVALID_ISLAND; //Already visited and not tagged with invalid island == a witness!
			break;
		}
		if( currentNode.index() == targetNode.index())
		{
			found = true;
			break;
		}

		mVisitedNodes.pushBack(TraversalState(currentNode, mVisitedNodes.size(), mVisitedNodes.size()-1, depth++));

		PX_ASSERT(mFastRoute[currentNode.index()].index() == PX_INVALID_NODE || isPathTo(currentNode, mFastRoute[currentNode.index()]));

		mIslandIds[currentNode.index()] = IG_INVALID_ISLAND;
		mVisitedState.set(currentNode.index());

		currentNode = mFastRoute[currentNode.index()];
	}
	while(currentNode.index() != PX_INVALID_NODE);

	for(PxU32 a = currentVisitedNodes; a < mVisitedNodes.size(); ++a)
	{
		const TraversalState& state = mVisitedNodes[a];
		mIslandIds[state.mNodeIndex.index()] = islandId;
	}

	if(!found)
	{
		for(PxU32 a = currentVisitedNodes; a < mVisitedNodes.size(); ++a)
		{
			const TraversalState& state = mVisitedNodes[a];
			mVisitedState.reset(state.mNodeIndex.index());
		}

		mVisitedNodes.forceSize_Unsafe(currentVisitedNodes);
	}
	return found;
}

bool IslandSim::findRoute(PxNodeIndex startNode, PxNodeIndex targetNode, IslandId islandId)
{
	//Firstly, traverse the fast path and tag up witnesses. TryFastPath can fail. In that case, no witnesses are left but this node is permitted to report
	//that it is still part of the island. Whichever node lost its fast path will be tagged as dirty and will be responsible for recovering the fast path
	//and tagging up the visited nodes
	if(mFastRoute[startNode.index()].index() != PX_INVALID_NODE)
	{
		if(tryFastPath(startNode, targetNode, islandId))
			return true;

		//Try fast path can either be successful or not. If it was successful, then we had a valid fast path cached and all nodes on that fast path were tagged
		//as witness nodes (visited and with a valid island ID). If the fast path was not successful, then no nodes were tagged as witnesses. 
		//Technically, we need to find a route to the root node but, as an optimization, we can simply return true from here with no witnesses added. 
		//Whichever node actually broke the "fast path" will also be on the list of dirty nodes and will be processed later. 
		//If that broken edge triggered an island separation, this node will be re-visited and added to that island, otherwise
		//the path to the root node will be re-established. The end result is the same - the island state is computed - this just saves us some work.
		//return true;
	}

	{
		//If we got here, there was no fast path. Therefore, we need to fall back on searching for the root node. This is optimized by using "hop counts".
		//These are per-node counts that indicate the expected number of hops from this node to the root node. These are lazily evaluated and updated
		//as new edges are formed or when traversals occur to re-establish islands. As a result, they may be inaccurate but they still serve the purpose
		//of guiding our search to minimize the chances of us doing an exhaustive search to find the root node.
		mIslandIds[startNode.index()] = IG_INVALID_ISLAND;
		TraversalState* startTraversal = &mVisitedNodes.pushBack(TraversalState(startNode, mVisitedNodes.size(), PX_INVALID_NODE, 0));
		mVisitedState.set(startNode.index());
		QueueElement element(startTraversal, mHopCounts[startNode.index()]);
		mPriorityQueue.push(element);

		do
		{
			const QueueElement currentQE = mPriorityQueue.pop();

			const TraversalState& currentState = *currentQE.mState;

			const Node& currentNode = mNodes[currentState.mNodeIndex.index()];

			EdgeInstanceIndex edge = currentNode.mFirstEdgeIndex;

			while(edge != IG_INVALID_EDGE)
			{
				const EdgeInstance& instance = mEdgeInstances[edge];
				{
					const PxNodeIndex nextIndex = mCpuData.mEdgeNodeIndices[edge ^ 1];

					//Static or kinematic nodes don't connect islands.
					if(nextIndex.index() != PX_INVALID_NODE && !mNodes[nextIndex.index()].isKinematic())
					{
						if(nextIndex.index() == targetNode.index())
						{
							unwindRoute(currentState.mCurrentIndex, nextIndex, 0, islandId);
							return true;
						}

						if(mVisitedState.test(nextIndex.index()))
						{
							//We already visited this node. This means that it's either in the priority queue already or we 
							//visited in on a previous pass. If it was visited on a previous pass, then it already knows what island it's in. 
							//We now need to test the island id to find out if this node knows the root.
							//If it has a valid root id, that id *is* our new root. We can guesstimate our hop count based on the node's properties
							
							const IslandId visitedIslandId = mIslandIds[nextIndex.index()];
							if(visitedIslandId != IG_INVALID_ISLAND)
							{
								//If we get here, we must have found a node that knows a route to our root node. It must not be a different island
								//because that would caused me to have been visited already because totally separate islands trigger a full traversal on 
								//the orphaned side.
								PX_ASSERT(visitedIslandId == islandId);
								unwindRoute(currentState.mCurrentIndex, nextIndex, mHopCounts[nextIndex.index()], islandId);
								return true;
							}
						}
						else
						{
							//This node has not been visited yet, so we need to push it into the stack and continue traversing
							TraversalState* state = &mVisitedNodes.pushBack(TraversalState(nextIndex, mVisitedNodes.size(), currentState.mCurrentIndex, currentState.mDepth+1));
							QueueElement qe(state, mHopCounts[nextIndex.index()]);
							mPriorityQueue.push(qe);
							mVisitedState.set(nextIndex.index());
							PX_ASSERT(mIslandIds[nextIndex.index()] == islandId);
							mIslandIds[nextIndex.index()] = IG_INVALID_ISLAND; //Flag as invalid island until we know whether we can find root or an island id.
						}
					}
				}

				edge = instance.mNextEdge;
			}
		}
		while(mPriorityQueue.size());

		return false;
	}
}

void IslandSim::processLostEdges(const PxArray<PxNodeIndex>& destroyedNodes, bool allowDeactivation, bool permitKinematicDeactivation, PxU32 dirtyNodeLimit)
{
	PX_UNUSED(dirtyNodeLimit);
	PX_PROFILE_ZONE("Basic.processLostEdges", mContextId);
	//At this point, all nodes and edges are activated. 

	//Bit map for visited
	mVisitedState.resizeAndClear(mNodes.size());

	//Reserve space on priority queue for at least 1024 nodes. It will resize if more memory is required during traversal.
	mPriorityQueue.reserve(1024);

	for (PxU32 i = 0; i < Edge::eEDGE_TYPE_COUNT; ++i)
		mIslandSplitEdges[i].reserve(1024);
	
	mVisitedNodes.reserve(mNodes.size()); //Make sure we have enough space for all nodes!

	const PxU32 nbDestroyedEdges = mDestroyedEdges.size();
	PX_UNUSED(nbDestroyedEdges);
	{
		PX_PROFILE_ZONE("Basic.removeEdgesFromIslands", mContextId);
		for (PxU32 a = 0; a < mDestroyedEdges.size(); ++a)
		{
			const EdgeIndex lostIndex = mDestroyedEdges[a];
			Edge& lostEdge = mEdges[lostIndex];

			if (lostEdge.isPendingDestroyed() && !lostEdge.isInDirtyList())
			{
				//Process this edge...
				if (!lostEdge.isReportOnlyDestroy() && lostEdge.isInserted())
				{
					const PxU32 index1 = mCpuData.mEdgeNodeIndices[mDestroyedEdges[a] * 2].index();
					const PxU32 index2 = mCpuData.mEdgeNodeIndices[mDestroyedEdges[a] * 2 + 1].index();

					IslandId islandId = IG_INVALID_ISLAND;
					if (index1 != PX_INVALID_NODE && index2 != PX_INVALID_NODE)
					{
						PX_ASSERT(mIslandIds[index1] == IG_INVALID_ISLAND || mIslandIds[index2] == IG_INVALID_ISLAND ||
							mIslandIds[index1] == mIslandIds[index2]);
						islandId = mIslandIds[index1] != IG_INVALID_ISLAND ? mIslandIds[index1] : mIslandIds[index2];
					}
					else if (index1 != PX_INVALID_NODE)
					{
						PX_ASSERT(index2 == PX_INVALID_NODE);
						Node& node = mNodes[index1];
						if (!node.isKinematic())
						{
							islandId = mIslandIds[index1];
							node.mStaticTouchCount--;
							//Island& island = mIslands[islandId];
							mIslandStaticTouchCount[islandId]--;
							//island.mStaticTouchCount--;
						}
					}
					else if (index2 != PX_INVALID_NODE)
					{
						PX_ASSERT(index1 == PX_INVALID_NODE);
						Node& node = mNodes[index2];
						if (!node.isKinematic())
						{
							islandId = mIslandIds[index2];
							node.mStaticTouchCount--;
							//Island& island = mIslands[islandId];
							mIslandStaticTouchCount[islandId]--;
							//island.mStaticTouchCount--;
						}
					}

					if (islandId != IG_INVALID_ISLAND)
					{
						//We need to remove this edge from the island
						Island& island = mIslands[islandId];
						removeEdgeFromIsland(island, lostIndex);
					}
				}

				lostEdge.clearInserted();
			}
		}
	}

	if (allowDeactivation)
	{
		PX_PROFILE_ZONE("Basic.findPathsAndBreakIslands", mContextId);

		//KS - process only this many dirty nodes, deferring future dirty nodes to subsequent frames. 
		//This means that it may take several frames for broken edges to trigger islands to completely break but this is better
		//than triggering large performance spikes.
#if IG_LIMIT_DIRTY_NODES
		PxBitMap::PxCircularIterator iter(mDirtyMap, mLastMapIndex);
		const PxU32 MaxCount = dirtyNodeLimit;// +10000000;
		PxU32 lastMapIndex = mLastMapIndex;
		PxU32 count = 0;
#else
		PxBitMap::Iterator iter(mDirtyMap);
#endif

		PxU32 dirtyIdx;

#if IG_LIMIT_DIRTY_NODES
		while ((dirtyIdx = iter.getNext()) != PxBitMap::PxCircularIterator::DONE
			&& (count++ < MaxCount)
#else
		while ((dirtyIdx = iter.getNext()) != PxBitMap::Iterator::DONE
#endif
			)
		{
#if IG_LIMIT_DIRTY_NODES
			lastMapIndex = dirtyIdx + 1;
#endif
			//Process dirty nodes. Figure out if we can make our way from the dirty node to the root.

			mPriorityQueue.clear(); //Clear the queue used for traversal
			mVisitedNodes.forceSize_Unsafe(0); //Clear the list of nodes in this island
			const PxNodeIndex dirtyNodeIndex(dirtyIdx);
			Node& dirtyNode = mNodes[dirtyNodeIndex.index()];

			//Check whether this node has already been touched. If it has been touched this frame, then its island state is reliable 
			//and we can just unclear the dirty flag on the body. If we were already visited, then the state should have already been confirmed in a 
			//previous pass.
			if (!dirtyNode.isKinematic() && !dirtyNode.isDeleted() && !mVisitedState.test(dirtyNodeIndex.index()))
			{
				//We haven't visited this node in our island repair passes yet, so we still need to process until we've hit a visited node or found
				//our root node. Note that, as soon as we hit a visited node that has already been processed in a previous pass, we know that we can rely
				//on its island information although the hop counts may not be optimal. It also indicates that this island was not broken immediately because
				//otherwise, the entire new sub-island would already have been visited and this node would have already had its new island state assigned.

				//Indicate that I've been visited

				const IslandId islandId = mIslandIds[dirtyNodeIndex.index()];
				const Island& findIsland = mIslands[islandId];

				const PxNodeIndex searchNode = findIsland.mRootNode;//The node that we're searching for!

				if (searchNode.index() != dirtyNodeIndex.index()) //If we are the root node, we don't need to do anything!
				{
					if (findRoute(dirtyNodeIndex, searchNode, islandId))
					{
						//We found the root node so let's let every visited node know that we found its root
						//and we can also update our hop counts because we recorded how many hops it took to reach this
						//node

						//We already filled in the path to the root/witness with accurate hop counts. Now we just need to fill in the estimates
						//for the remaining nodes and re-define their islandIds. We approximate their path to the root by just routing them through
						//the route we already found.

						//This loop works because mVisitedNodes are recorded in the order they were visited and we already filled in the critical path
						//so the remainder of the paths will just fork from that path.

						//Verify state (that we can see the root from this node)...

#if IG_SANITY_CHECKS
						PX_ASSERT(canFindRoot(dirtyNodeIndex, searchNode, NULL)); //Verify that we found the connection
#endif

						for (PxU32 b = 0; b < mVisitedNodes.size(); ++b)
						{
							TraversalState& state = mVisitedNodes[b];
							if (mIslandIds[state.mNodeIndex.index()] == IG_INVALID_ISLAND)
							{
								mHopCounts[state.mNodeIndex.index()] = mHopCounts[mVisitedNodes[state.mPrevIndex].mNodeIndex.index()] + 1;
								mFastRoute[state.mNodeIndex.index()] = mVisitedNodes[state.mPrevIndex].mNodeIndex;
								mIslandIds[state.mNodeIndex.index()] = islandId;
							}
						}
					}
					else
					{
						//If I traversed and could not find the root node, then I have established a new island. In this island, I am the root node
						//and I will point all my nodes towards me. Furthermore, I have established how many steps it took to reach all nodes in my island

						//OK. We need to separate the islands. We have a list of nodes that are part of the new island (mVisitedNodes) and we know that the 
						//first node in that list is the root node.


						//OK, we need to remove all these actors from their current island, then add them to the new island...

						Island& oldIsland = mIslands[islandId];
						//We can just unpick these nodes from the island because they do not contain the root node (if they did, then we wouldn't be
						//removing this node from the island at all). The only challenge is if we need to remove the last node. In that case
						//we need to re-establish the new last node in the island but perhaps the simplest way to do that would be to traverse
						//the island to establish the last node again

#if IG_SANITY_CHECKS
						PX_ASSERT(!canFindRoot(dirtyNodeIndex, searchNode, NULL));
#endif

						PxU32 totalStaticTouchCount = 0;
						PxU32 nodeCount[Node::eTYPE_COUNT];
						for (PxU32 t = 0; t < Node::eTYPE_COUNT; ++t)
						{
							nodeCount[t] = 0;
						}

						for (PxU32 t = 0; t < Edge::eEDGE_TYPE_COUNT; ++t)
						{
							mIslandSplitEdges[t].forceSize_Unsafe(0);
						}

						//NodeIndex lastIndex = oldIsland.mLastNode;

						//nodeCount[node.mType] = 1;

						for (PxU32 a = 0; a < mVisitedNodes.size(); ++a)
						{
							const PxNodeIndex index = mVisitedNodes[a].mNodeIndex;
							Node& node = mNodes[index.index()];

							if (node.mNextNode.index() != PX_INVALID_NODE)
								mNodes[node.mNextNode.index()].mPrevNode = node.mPrevNode;
							else
								oldIsland.mLastNode = node.mPrevNode;
							if (node.mPrevNode.index() != PX_INVALID_NODE)
								mNodes[node.mPrevNode.index()].mNextNode = node.mNextNode;

							nodeCount[node.mType]++;

							node.mNextNode.setIndices(PX_INVALID_NODE);
							node.mPrevNode.setIndices(PX_INVALID_NODE);

							PX_ASSERT(mNodes[oldIsland.mLastNode.index()].mNextNode.index() == PX_INVALID_NODE);

							totalStaticTouchCount += node.mStaticTouchCount;

							EdgeInstanceIndex idx = node.mFirstEdgeIndex;

							while (idx != IG_INVALID_EDGE)
							{
								const EdgeInstance& instance = mEdgeInstances[idx];
								const EdgeIndex edgeIndex = idx / 2;
								const Edge& edge = mEdges[edgeIndex];

								//Only split the island if we're processing the first node or if the first node is infinte-mass
								if (!(idx & 1) || (mCpuData.mEdgeNodeIndices[idx & (~1)].index() == PX_INVALID_NODE || mNodes[mCpuData.mEdgeNodeIndices[idx & (~1)].index()].isKinematic()))
								{
									//We will remove this edge from the island...
									mIslandSplitEdges[edge.mEdgeType].pushBack(edgeIndex);

									removeEdgeFromIsland(oldIsland, edgeIndex);
								}
								idx = instance.mNextEdge;
							}
						}

						//oldIsland.mStaticTouchCount -= totalStaticTouchCount;
						mIslandStaticTouchCount[islandId] -= totalStaticTouchCount;

						for (PxU32 i = 0; i < Node::eTYPE_COUNT; ++i)
						{
							PX_ASSERT(nodeCount[i] <= oldIsland.mNodeCount[i]);
							oldIsland.mNodeCount[i] -= nodeCount[i];
						}

						//Now add all these nodes to the new island

						//(1) Create the new island...
						const IslandId newIslandHandle = mIslandHandles.getHandle();
						/*if(newIslandHandle == mIslands.capacity())
						{
						mIslands.reserve(2*mIslands.capacity() + 1);
						}*/
						mIslands.resize(PxMax(newIslandHandle + 1, mIslands.size()));
						mIslandStaticTouchCount.resize(PxMax(newIslandHandle + 1, mIslandStaticTouchCount.size()));
						Island& newIsland = mIslands[newIslandHandle];

						if (mIslandAwake.test(islandId))
						{
							newIsland.mActiveIndex = mActiveIslands.size();
							mActiveIslands.pushBack(newIslandHandle);
							mIslandAwake.growAndSet(newIslandHandle); //Separated island, so it should be awake
						}
						else
						{
							mIslandAwake.growAndReset(newIslandHandle);
						}

						newIsland.mRootNode = dirtyNodeIndex;
						mHopCounts[dirtyNodeIndex.index()] = 0;
						mIslandIds[dirtyNodeIndex.index()] = newIslandHandle;
						//newIsland.mTotalSize = mVisitedNodes.size();

						mNodes[dirtyNodeIndex.index()].mPrevNode.setIndices(PX_INVALID_NODE); //First node so doesn't have a preceding node
						mFastRoute[dirtyNodeIndex.index()].setIndices(PX_INVALID_NODE);

						for (PxU32 i = 0; i < Node::eTYPE_COUNT; ++i)
							nodeCount[i] = 0;

						nodeCount[dirtyNode.mType] = 1;

						for (PxU32 a = 1; a < mVisitedNodes.size(); ++a)
						{
							const PxNodeIndex index = mVisitedNodes[a].mNodeIndex;
							Node& thisNode = mNodes[index.index()];
							const PxNodeIndex prevNodeIndex = mVisitedNodes[a - 1].mNodeIndex;
							thisNode.mPrevNode = prevNodeIndex;
							mNodes[prevNodeIndex.index()].mNextNode = index;
							nodeCount[thisNode.mType]++;
							mIslandIds[index.index()] = newIslandHandle;
							mHopCounts[index.index()] = mVisitedNodes[a].mDepth; //How many hops to root
							mFastRoute[index.index()] = mVisitedNodes[mVisitedNodes[a].mPrevIndex].mNodeIndex;
						}

						for (PxU32 i = 0; i < Node::eTYPE_COUNT; ++i)
							newIsland.mNodeCount[i] = nodeCount[i];

						//Last node in the island
						const PxNodeIndex lastIndex = mVisitedNodes[mVisitedNodes.size() - 1].mNodeIndex;
						mNodes[lastIndex.index()].mNextNode.setIndices(PX_INVALID_NODE);
						newIsland.mLastNode = lastIndex;
						//newIsland.mStaticTouchCount = totalStaticTouchCount;
						mIslandStaticTouchCount[newIslandHandle] = totalStaticTouchCount;

						PX_ASSERT(mNodes[newIsland.mLastNode.index()].mNextNode.index() == PX_INVALID_NODE);

						for (PxU32 j = 0; j < IG::Edge::eEDGE_TYPE_COUNT; ++j)
						{
							PxArray<EdgeIndex>& splitEdges = mIslandSplitEdges[j];
							const PxU32 splitEdgeSize = splitEdges.size();
							if (splitEdgeSize)
							{
								splitEdges.pushBack(IG_INVALID_EDGE); //Push in a dummy invalid edge to complete the connectivity
								mEdges[splitEdges[0]].mNextIslandEdge = splitEdges[1];
								for (PxU32 a = 1; a < splitEdgeSize; ++a)
								{
									const EdgeIndex edgeIndex = splitEdges[a];
									Edge& edge = mEdges[edgeIndex];
									edge.mNextIslandEdge = splitEdges[a + 1];
									edge.mPrevIslandEdge = splitEdges[a - 1];
								}

								newIsland.mFirstEdge[j] = splitEdges[0];
								newIsland.mLastEdge[j] = splitEdges[splitEdgeSize - 1];
								newIsland.mEdgeCount[j] = splitEdgeSize;
							}
						}
					}
				}
			}
			dirtyNode.clearDirty();
#if IG_LIMIT_DIRTY_NODES
			mDirtyMap.reset(dirtyIdx);
#endif
		}

#if IG_LIMIT_DIRTY_NODES
		mLastMapIndex = lastMapIndex;
		if (count < MaxCount)
			mLastMapIndex = 0;
#else
		mDirtyMap.clear();
#endif

		//mDirtyNodes.forceSize_Unsafe(0);
	}

	{
		PX_PROFILE_ZONE("Basic.clearDestroyedEdges", mContextId);
		//Now process the lost edges...
		for (PxU32 a = 0; a < mDestroyedEdges.size(); ++a)
		{
			//Process these destroyed edges. Recompute island information. Update the islands and hop counters accordingly
			const EdgeIndex index = mDestroyedEdges[a];

			Edge& edge = mEdges[index];
			if (edge.isPendingDestroyed())
			{
				PartitionEdge* pEdge = mGpuData ? mGpuData->mFirstPartitionEdges[index] : NULL;
				if (pEdge)
				{
					mGpuData->mDestroyedPartitionEdges.pushBack(pEdge);
					mGpuData->mFirstPartitionEdges[index] = NULL; //Force first partition edge to NULL to ensure we don't have a clash
				}
				if (edge.isActive())
				{
					removeEdgeFromActivatingList(index); //TODO - can we remove this call? Can we handle this elsewhere, e.g. when destroying the nodes...
					mActiveEdgeCount[edge.mEdgeType]--;
				}

				edge = Edge(); //Reset edge
				if(mGpuData)
					mGpuData->mActiveContactEdges.growAndReset(index);
			}
		}

		mDestroyedEdges.forceSize_Unsafe(0);
	}

	{
		PX_PROFILE_ZONE("Basic.clearDestroyedNodes", mContextId);

		for (PxU32 a = 0; a < destroyedNodes.size(); ++a)
		{
			const PxNodeIndex nodeIndex = destroyedNodes[a];
			const IslandId islandId = mIslandIds[nodeIndex.index()];
			Node& node = mNodes[nodeIndex.index()];
			if (islandId != IG_INVALID_ISLAND)
			{
				Island& island = mIslands[islandId];

				removeNodeFromIsland(island, nodeIndex);

				mIslandIds[nodeIndex.index()] = IG_INVALID_ISLAND;

				PxU32 nodeCountTotal = 0;
				for (PxU32 t = 0; t < Node::eTYPE_COUNT; ++t)
				{
					nodeCountTotal += island.mNodeCount[t];
				}

				if (nodeCountTotal == 0)
				{
					mIslandHandles.freeHandle(islandId);
					if (island.mActiveIndex != IG_INVALID_ISLAND)
					{
						const IslandId replaceId = mActiveIslands[mActiveIslands.size() - 1];
						Island& replaceIsland = mIslands[replaceId];
						replaceIsland.mActiveIndex = island.mActiveIndex;
						mActiveIslands[island.mActiveIndex] = replaceId;
						mActiveIslands.forceSize_Unsafe(mActiveIslands.size() - 1);
						island.mActiveIndex = IG_INVALID_ISLAND;
						//island.mStaticTouchCount -= node.mStaticTouchCount; //Remove the static touch count from the island
						mIslandStaticTouchCount[islandId] -= node.mStaticTouchCount;
					}
					mIslandAwake.reset(islandId);
					island.mLastNode.setIndices(PX_INVALID_NODE);
					island.mRootNode.setIndices(PX_INVALID_NODE);
					island.mActiveIndex = IG_INVALID_ISLAND;
				}
			}

			if (node.isKinematic())
			{
				if (mActiveNodeIndex[nodeIndex.index()] != PX_INVALID_NODE)
				{
					//Remove from the active kinematics list...
					markKinematicInactive(nodeIndex);
				}
			}
			else
			{
				if (mActiveNodeIndex[nodeIndex.index()] != PX_INVALID_NODE)
				{
					markInactive(nodeIndex);
				}
			}

			//node.reset();
			node.mFlags |= Node::eDELETED;
		}
	}
	//Now we need to produce the list of active edges and nodes!!!

	//If we get here, we have a list of active islands. From this, we need to iterate over all active islands and establish if that island
	//can, in fact, go to sleep. In order to become deactivated, all nodes in the island must be ready for sleeping...

	if (allowDeactivation)
	{
		PX_PROFILE_ZONE("Basic.deactivation", mContextId);
		for (PxU32 a = 0; a < mActiveIslands.size(); a++)
		{
			const IslandId islandId = mActiveIslands[a];

			mIslandAwake.reset(islandId);
		}

		//Loop over the active kinematic nodes and tag all islands touched by active kinematics as awake
		for (PxU32 a = mActiveKinematicNodes.size(); a > 0; --a)
		{
			const PxNodeIndex kinematicIndex = mActiveKinematicNodes[a - 1];

			Node& kinematicNode = mNodes[kinematicIndex.index()];

			if (kinematicNode.isReadyForSleeping())
			{
				if (permitKinematicDeactivation)
				{
					kinematicNode.clearActive();
					markKinematicInactive(kinematicIndex);
				}
			}
			else //if(!kinematicNode.isReadyForSleeping())
			{
				//KS - if kinematic is active, then wake up all islands the kinematic is touching
				EdgeInstanceIndex edgeId = kinematicNode.mFirstEdgeIndex;
				while (edgeId != IG_INVALID_EDGE)
				{
					const EdgeInstance& instance = mEdgeInstances[edgeId];
					//Edge& edge = mEdges[edgeId/2];
					//Only wake up islands if a connection was present
					//if(edge.isConnected())
					{
						PxNodeIndex outNode = mCpuData.mEdgeNodeIndices[edgeId ^ 1];
						if (outNode.index() != PX_INVALID_NODE)
						{
							IslandId islandId = mIslandIds[outNode.index()];
							if (islandId != IG_INVALID_ISLAND)
							{
								mIslandAwake.set(islandId);
								PX_ASSERT(mIslands[islandId].mActiveIndex != IG_INVALID_ISLAND);
							}
						}
					}
					edgeId = instance.mNextEdge;
				}
			}
		}

		for (PxU32 a = mActiveIslands.size(); a > 0; --a)
		{
			const IslandId islandId = mActiveIslands[a - 1];

			const Island& island = mIslands[islandId];

			bool canDeactivate = !mIslandAwake.test(islandId);
			mIslandAwake.set(islandId);

			//If it was touched by an active kinematic in the above loop, we can't deactivate it.
			//Therefore, no point in testing the nodes in the island. They must remain awake
			if (canDeactivate)
			{
				PxNodeIndex nodeId = island.mRootNode;
				while (nodeId.index() != PX_INVALID_NODE)
				{
					Node& node = mNodes[nodeId.index()];
					if (!node.isReadyForSleeping())
					{
						canDeactivate = false;
						break;
					}
					nodeId = node.mNextNode;
				}
				if (canDeactivate)
				{
					//If all nodes in this island are ready for sleeping and there were no active 
					//kinematics interacting with the any bodies in the island, we can deactivate the island.
					deactivateIsland(islandId);
				}
			}
		}
	}

	{
		PX_PROFILE_ZONE("Basic.resetDirtyEdges", mContextId);
		for (PxU32 i = 0; i < Edge::eEDGE_TYPE_COUNT; ++i)
		{
			for (PxU32 a = 0; a < mDirtyEdges[i].size(); ++a)
			{
				Edge& edge = mEdges[mDirtyEdges[i][a]];
				edge.clearInDirtyList();
			}
			mDirtyEdges[i].clear(); //All new edges processed
		}
	}
}

IslandId IslandSim::mergeIslands(IslandId island0, IslandId island1, PxNodeIndex node0, PxNodeIndex node1)
{
	Island& is0 = mIslands[island0];
	Island& is1 = mIslands[island1];

	//We defer this process and do it later instead. That way, if we have some pathological 
	//case where multiple islands get merged repeatedly, we don't end up repeatedly remapping all the nodes in those islands 
	//to their new island. Instead, we just choose the largest island and remap the smaller island to that.

	PxU32 totalSize0 = 0;
	PxU32 totalSize1 = 0;

	for (PxU32 i = 0; i < Node::eTYPE_COUNT; ++i)
	{
		totalSize0 += is0.mNodeCount[i];
		totalSize1 += is1.mNodeCount[i];
	}
	if(totalSize0 > totalSize1)
	{
		mergeIslandsInternal(is0, is1, island0, island1, node0, node1);
		mIslandAwake.reset(island1);
		mIslandHandles.freeHandle(island1);
		mFastRoute[node1.index()] = node0;
		return island0;
	}
	else
	{
		mergeIslandsInternal(is1, is0, island1, island0, node1, node0);
		mIslandAwake.reset(island0);
		mIslandHandles.freeHandle(island0);
		mFastRoute[node0.index()] = node1;
		return island1;
	}
}

bool IslandSim::checkInternalConsistency() const
{
	//Loop over islands, confirming that the island data is consistent...
	//Really expensive. Turn off unless investigating some random issue...
#if 0
	for (PxU32 a = 0; a < mIslands.size(); ++a)
	{
		const Island& island = mIslands[a];

		PxU32 expectedNodeCount = 0;
		for (PxU32 t = 0; t < Node::eTYPE_COUNT; ++t)
		{
			expectedNodeCount += island.mNodeCount[t];
		}
		bool metLastNode = expectedNodeCount == 0;

		PxNodeIndex nodeId = island.mRootNode;

		while (nodeId.index() != PX_INVALID_NODE)
		{
			PX_ASSERT(mIslandIds[nodeId.index()] == a);

			if (nodeId.index() == island.mLastNode.index())
			{
				metLastNode = true;
				PX_ASSERT(mNodes[nodeId.index()].mNextNode.index() == PX_INVALID_NODE);
			}

			--expectedNodeCount;

			nodeId = mNodes[nodeId.index()].mNextNode;
		}

		PX_ASSERT(expectedNodeCount == 0);
		PX_ASSERT(metLastNode);
	}
#endif

	return true;
}

void IslandSim::mergeIslandsInternal(Island& island0, Island& island1, IslandId islandId0, IslandId islandId1, PxNodeIndex nodeIndex0, PxNodeIndex nodeIndex1)
{	
#if PX_ENABLE_ASSERTS
	PxU32 island0Size = 0;
	PxU32 island1Size = 0;
	for(PxU32 nodeType = 0; nodeType < Node::eTYPE_COUNT; ++nodeType)
	{
		island0Size += island0.mNodeCount[nodeType];
		island1Size += island1.mNodeCount[nodeType];
	}
#endif
	PX_ASSERT(island0Size >= island1Size); //We only ever merge the smaller island to the larger island
	//Stage 1 - we need to move all the nodes across to the new island ID (i.e. write all their new island indices, move them to the 
	//island and then also update their estimated hop counts to the root. As we don't want to do a full traversal at this point,
	//instead, we estimate based on the route from the node to their previous root and then from that root to the new connection
	//between the 2 islands. This is probably a very indirect route but it will be refined later.

	//In this case, island1 is subsumed by island0

	//It takes mHopCounts[nodeIndex1] to get from node1 to its old root. It takes mHopCounts[nodeIndex0] to get from nodeIndex0 to the new root
	//and it takes 1 extra hop to go from node1 to node0. Therefore, a sub-optimal route can be planned going via the old root node that should take
	//mHopCounts[nodeIndex0] + mHopCounts[nodeIndex1] + 1 + mHopCounts[nodeIndex] to travel from any arbitrary node (nodeIndex) in island1 to the root
	//of island2.

	const PxU32 extraPath = mHopCounts[nodeIndex0.index()] + mHopCounts[nodeIndex1.index()] + 1;

	PxNodeIndex islandNode = island1.mRootNode;
	while(islandNode.index() != PX_INVALID_NODE)
	{
		mHopCounts[islandNode.index()] += extraPath;
		mIslandIds[islandNode.index()] = islandId0;

		//mFastRoute[islandNode] = PX_INVALID_NODE;
		
		Node& node = mNodes[islandNode.index()];
		islandNode = node.mNextNode;				
	}

	//Now fill in the hop count for node1, which is directly connected to node0.
	mHopCounts[nodeIndex1.index()] = mHopCounts[nodeIndex0.index()] + 1;
	Node& lastNode = mNodes[island0.mLastNode.index()];
	Node& firstNode = mNodes[island1.mRootNode.index()];
	PX_ASSERT(lastNode.mNextNode.index() == PX_INVALID_NODE);
	PX_ASSERT(firstNode.mPrevNode.index() == PX_INVALID_NODE);
	PX_ASSERT(island1.mRootNode.index() != island0.mLastNode.index());

	PX_ASSERT(mNodes[island0.mLastNode.index()].mNextNode.index() == PX_INVALID_NODE);
	PX_ASSERT(mNodes[island1.mLastNode.index()].mNextNode.index() == PX_INVALID_NODE);

	PX_ASSERT(mIslandIds[island0.mLastNode.index()] == islandId0);

	lastNode.mNextNode = island1.mRootNode;
	firstNode.mPrevNode = island0.mLastNode;

	island0.mLastNode = island1.mLastNode;
	//island0.mStaticTouchCount += island1.mStaticTouchCount;
	mIslandStaticTouchCount[islandId0] += mIslandStaticTouchCount[islandId1];

	//Merge the edge list for the islands...
	for(PxU32 a = 0; a < IG::Edge::eEDGE_TYPE_COUNT; ++a)
	{
		if(island0.mLastEdge[a] != IG_INVALID_EDGE)
		{
			PX_ASSERT(mEdges[island0.mLastEdge[a]].mNextIslandEdge == IG_INVALID_EDGE);
			mEdges[island0.mLastEdge[a]].mNextIslandEdge = island1.mFirstEdge[a];
		}
		else
		{
			PX_ASSERT(island0.mFirstEdge[a] == IG_INVALID_EDGE);
			island0.mFirstEdge[a] = island1.mFirstEdge[a];
		}
		if(island1.mFirstEdge[a] != IG_INVALID_EDGE)
		{
			PX_ASSERT(mEdges[island1.mFirstEdge[a]].mPrevIslandEdge == IG_INVALID_EDGE);
			mEdges[island1.mFirstEdge[a]].mPrevIslandEdge = island0.mLastEdge[a];
			island0.mLastEdge[a] = island1.mLastEdge[a];
		}

		island0.mEdgeCount[a] += island1.mEdgeCount[a];
		island1.mFirstEdge[a] = IG_INVALID_EDGE;
		island1.mLastEdge[a] = IG_INVALID_EDGE;
		island1.mEdgeCount[a] = 0;
	}

	for (PxU32 a = 0; a < IG::Node::eTYPE_COUNT; ++a)
	{
		island0.mNodeCount[a] += island1.mNodeCount[a];
		island1.mNodeCount[a] = 0;
	}

	island1.mLastNode.setIndices(PX_INVALID_NODE);
	island1.mRootNode.setIndices(PX_INVALID_NODE);
	
	mIslandStaticTouchCount[islandId1] = 0;
	//island1.mStaticTouchCount = 0;
	
	//Remove from active island list
	if(island1.mActiveIndex != IG_INVALID_ISLAND)
		markIslandInactive(islandId1);
}

void IslandSim::removeEdgeFromActivatingList(EdgeIndex index)
{
	Edge& edge = mEdges[index];

	if (edge.mEdgeState & Edge::eACTIVATING)
	{
		for (PxU32 a = 0, count = mActivatedEdges[edge.mEdgeType].size(); a < count; a++)
		{
			if (mActivatedEdges[edge.mEdgeType][a] == index)
			{
				mActivatedEdges[edge.mEdgeType].replaceWithLast(a);
				break;
			}
		}

		edge.mEdgeState &= (~Edge::eACTIVATING);
	}

	const PxNodeIndex nodeIndex1 = mCpuData.mEdgeNodeIndices[index * 2];
	const PxNodeIndex nodeIndex2 = mCpuData.mEdgeNodeIndices[index * 2 + 1];

	if (nodeIndex1.isValid() && nodeIndex2.isValid())
	{
		mNodes[nodeIndex1.index()].mActiveRefCount--;
		mNodes[nodeIndex2.index()].mActiveRefCount--;
	}

	if(mGpuData && edge.mEdgeType == Edge::eCONTACT_MANAGER)
		mGpuData->mActiveContactEdges.reset(index);
}

void IslandSim::setKinematic(PxNodeIndex nodeIndex)
{
	Node& node = mNodes[nodeIndex.index()];

	if(!node.isKinematic())
	{
		//Transition from dynamic to kinematic:
		//(1) Remove this node from the island
		//(2) Remove this node from the active node list
		//(3) If active or referenced, add it to the active kinematic list
		//(4) Tag the node as kinematic
		//External code will re-filter interactions and lost touches will be reported

		const IslandId islandId = mIslandIds[nodeIndex.index()];
		PX_ASSERT(islandId != IG_INVALID_ISLAND);

		Island& island = mIslands[islandId];

		mIslandIds[nodeIndex.index()] = IG_INVALID_ISLAND;

		removeNodeFromIsland(island, nodeIndex);

		const bool isActive = node.isActive()!=0;

		if (isActive)
		{
			//Remove from active list...
			markInactive(nodeIndex);
		}
		else if (node.isActivating())
		{
			//Remove from activating list...
			node.clearActivating();
			PX_ASSERT(mActivatingNodes[mActiveNodeIndex[nodeIndex.index()]].index() == nodeIndex.index());

			const PxNodeIndex replaceIndex = mActivatingNodes[mActivatingNodes.size() - 1];
			mActiveNodeIndex[replaceIndex.index()] = mActiveNodeIndex[nodeIndex.index()];
			mActivatingNodes[mActiveNodeIndex[nodeIndex.index()]] = replaceIndex;
			mActivatingNodes.forceSize_Unsafe(mActivatingNodes.size() - 1);
			mActiveNodeIndex[nodeIndex.index()] = PX_INVALID_NODE;
		}

		node.setKinematicFlag();

		node.clearActive();

		if (/*isActive || */node.mActiveRefCount != 0)
		{
			//Add to active kinematic list...
			PX_ASSERT(mActiveNodeIndex[nodeIndex.index()] == PX_INVALID_NODE);

			mActiveNodeIndex[nodeIndex.index()] = mActivatingNodes.size();
			mActivatingNodes.pushBack(nodeIndex);
			node.setActivating();
		}

		{
			//This node was potentially in an island with other bodies. We need to force an island recomputation in case the
			//islands became broken due to losing this connection. Same rules as losing a contact, we just 
			//tag the nodes directly connected to the lost edge as "dirty" and force an island recomputation if 
			//it resulted in lost connections
			EdgeInstanceIndex edgeId = node.mFirstEdgeIndex;
			while(edgeId != IG_INVALID_EDGE)
			{
				const EdgeInstance& instance = mEdgeInstances[edgeId];
				const EdgeInstanceIndex nextId = instance.mNextEdge;

				const PxU32 idx = edgeId/2;
				IG::Edge& edge = mEdges[edgeId/2];

				removeEdgeFromIsland(island, idx);

				removeConnectionInternal(idx);
				removeConnectionFromGraph(idx);

				edge.clearInserted();

				if (edge.isActive())
				{
					removeEdgeFromActivatingList(idx);
					edge.deactivateEdge();
					mActiveEdgeCount[edge.mEdgeType]--;
					mDeactivatingEdges[edge.mEdgeType].pushBack(idx);
				}

				if(!edge.isPendingDestroyed())
				{
					if(!edge.isInDirtyList())
					{
						PX_ASSERT(!contains(mDirtyEdges[edge.mEdgeType], idx));
						mDirtyEdges[edge.mEdgeType].pushBack(idx);
						edge.markInDirtyList();
					}
				}
				else
				{
					edge.setReportOnlyDestroy();
				}

				edgeId = nextId;
			}
		}

		PxU32 newNodeCount = 0;
		for(PxU32 i = 0; i < Node::eTYPE_COUNT; ++i)
			newNodeCount += island.mNodeCount[i];

		if(newNodeCount == 0)
		{
			// If this island is empty after having removed the edges of the node we've just set to kinematic
			// we invalidate all edges and set the island to inactive
			for(PxU32 a = 0; a < Edge::eEDGE_TYPE_COUNT; ++a)
			{
				island.mFirstEdge[a] = island.mLastEdge[a] = IG_INVALID_EDGE;
				island.mEdgeCount[a] = 0;
				mIslandStaticTouchCount[islandId] = 0;
				//island.mStaticTouchCount = 0;
			}

			if(island.mActiveIndex != IG_INVALID_ISLAND)
			{
				markIslandInactive(islandId);
			}

			mIslandAwake.reset(islandId);
			mIslandHandles.freeHandle(islandId);
		}
	}
}

void IslandSim::setDynamic(PxNodeIndex nodeIndex)
{
	//(1) Remove all edges involving this node from all islands they may be in
	//(2) Mark all edges as "new" edges - let island gen re-process them!
	//(3) Remove this node from the active kinematic list
	//(4) Add this node to the active dynamic list (if it is active)
	//(5) Mark node as dynamic
	
	Node& node = mNodes[nodeIndex.index()];

	if(node.isKinematic())
	{
		//EdgeInstanceIndex edgeIndex = node.mFirstEdgeIndex;

		EdgeInstanceIndex edgeId = node.mFirstEdgeIndex;
		while(edgeId != IG_INVALID_EDGE)
		{
			const EdgeInstance& instance = mEdgeInstances[edgeId];
			const EdgeInstanceIndex nextId = instance.mNextEdge;

			const PxNodeIndex otherNode = mCpuData.mEdgeNodeIndices[edgeId^1];

			const PxU32 idx = edgeId/2;
			IG::Edge& edge = mEdges[edgeId/2];

			if(!otherNode.isStaticBody())
			{
				const IslandId islandId = mIslandIds[otherNode.index()];
				if(islandId != IG_INVALID_ISLAND)
					removeEdgeFromIsland(mIslands[islandId], idx);
			}

			removeConnectionInternal(idx);
			removeConnectionFromGraph(idx);

			edge.clearInserted();
			if (edge.isActive())
			{
				edge.deactivateEdge();
				removeEdgeFromActivatingList(idx);
				mActiveEdgeCount[edge.mEdgeType]--;
			}
			
			if(!edge.isPendingDestroyed())
			{
				if(!edge.isInDirtyList())
				{
					PX_ASSERT(!contains(mDirtyEdges[edge.mEdgeType], idx));
					mDirtyEdges[edge.mEdgeType].pushBack(idx);
					edge.markInDirtyList();
				}
			}
			else
			{
				edge.setReportOnlyDestroy();
			}

			edgeId = nextId;
		}

		if(!node.isActivating() && mActiveNodeIndex[nodeIndex.index()] != PX_INVALID_NODE)
		{
			//Remove from active kinematic list, add to active dynamic list
			const PxU32 oldRefCount = node.mActiveRefCount;
			node.mActiveRefCount = 0;
			markKinematicInactive(nodeIndex);
			node.mActiveRefCount = oldRefCount;
		}

		node.clearKinematicFlag();

		//Create an island for this node. If there are any edges affecting this node, they will have been marked as 
		//"new" and will be processed next island update.
		{
			const IslandId islandHandle = mIslandHandles.getHandle();
			
			if(islandHandle == mIslands.capacity())
			{
				const PxU32 newCapacity = 2*mIslands.capacity()+1;
				mIslands.reserve(newCapacity);
				mIslandAwake.resize(newCapacity);
				mIslandStaticTouchCount.resize(newCapacity);
			}
			mIslandAwake.reset(islandHandle);
			mIslands.resize(PxMax(islandHandle+1, mIslands.size()));
			mIslandStaticTouchCount.resize(PxMax(islandHandle + 1, mIslands.size()));
			Island& island = mIslands[islandHandle];
			island.mLastNode = island.mRootNode = nodeIndex;
			PX_ASSERT(mNodes[nodeIndex.index()].mNextNode.index() == PX_INVALID_NODE);
			island.mNodeCount[node.mType] = 1;
			mIslandIds[nodeIndex.index()] = islandHandle;
			mIslandStaticTouchCount[islandHandle] = 0;

			if(node.isActive())
			{
				node.clearActive();

				activateNode(nodeIndex);
			}
		}
	}
}
