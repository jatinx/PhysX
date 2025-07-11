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

#ifdef RENDER_SNIPPET

#include "PxPhysicsAPI.h"
#include "cudamanager/PxCudaContext.h"
#include "cudamanager/PxCudaContextManager.h"

#include "../snippetrender/SnippetRender.h"
#include "../snippetrender/SnippetCamera.h"

#define hipSuccess 0
#define SHOW_SOLID_SDF_SLICE 0
#define IDX(i, j, k, offset) ((i) + dimX * ((j) + dimY * ((k) + dimZ * (offset))))
using namespace physx;

extern void initPhysics(bool interactive);
extern void stepPhysics(bool interactive);	
extern void cleanupPhysics(bool interactive);
extern void keyPress(unsigned char key, const PxTransform& camera);
extern PxPBDParticleSystem* getParticleSystem();
extern PxParticleAndDiffuseBuffer* getParticleBuffer();

extern int getNumDiffuseParticles();


namespace
{
Snippets::Camera* sCamera;

Snippets::SharedGLBuffer sPosBuffer;
Snippets::SharedGLBuffer sDiffusePosLifeBuffer;

void onBeforeRenderParticles()
{	
}

void renderParticles()
{

	PxPBDParticleSystem* particleSystem = getParticleSystem();
	if (particleSystem)
	{
		PxParticleAndDiffuseBuffer* userBuffer = getParticleBuffer();
		PxVec4* positions = userBuffer->getPositionInvMasses();
		PxVec4* diffusePositions = userBuffer->getDiffusePositionLifeTime();

		const PxU32 numParticles = userBuffer->getNbActiveParticles();
		const PxU32 numDiffuseParticles = userBuffer->getNbActiveDiffuseParticles();

		PxScene* scene;
		PxGetPhysics().getScenes(&scene, 1);
		PxCudaContextManager* cudaContextManager = scene->getCudaContextManager();

		cudaContextManager->acquireContext();

		PxCudaContext* cudaContext = cudaContextManager->getCudaContext();
		cudaContext->memcpyDtoH(sPosBuffer.map(), hipDeviceptr_t(positions), sizeof(PxVec4) * numParticles);
		cudaContext->memcpyDtoH(sDiffusePosLifeBuffer.map(), hipDeviceptr_t(diffusePositions), sizeof(PxVec4) * numDiffuseParticles);

		cudaContextManager->releaseContext();

#if SHOW_SOLID_SDF_SLICE
		particleSystem->copySparseGridData(sSparseGridSolidSDFBufferD, PxSparseGridDataFlag::eGRIDCELL_SOLID_GRADIENT_AND_SDF);
#endif
	}

	sPosBuffer.unmap();
	sDiffusePosLifeBuffer.unmap();
	PxVec3 color(0.5f, 0.5f, 1);
	Snippets::DrawPoints(sPosBuffer.vbo, sPosBuffer.size / sizeof(PxVec4), color, 2.f);

	PxParticleAndDiffuseBuffer* userBuffer = getParticleBuffer();
	if (userBuffer)
	{
		const PxU32 numActiveDiffuseParticles = userBuffer->getNbActiveDiffuseParticles();

		//printf("NumActiveDiffuse = %i\n", numActiveDiffuseParticles);

		if (numActiveDiffuseParticles > 0)
		{
			PxVec3 colorDiffuseParticles(1, 1, 1);
			Snippets::DrawPoints(sDiffusePosLifeBuffer.vbo, numActiveDiffuseParticles, colorDiffuseParticles, 2.f);
		}
	}
	
	Snippets::DrawFrame(PxVec3(0, 0, 0));
}

void allocParticleBuffers()
{
	PxScene* scene;
	PxGetPhysics().getScenes(&scene, 1);
	PxCudaContextManager* cudaContextManager = scene->getCudaContextManager();
	if (cudaContextManager)
	{
		PxParticleAndDiffuseBuffer* userBuffer = getParticleBuffer();

		const PxU32 maxParticles = userBuffer->getMaxParticles();
		const PxU32 maxDiffuseParticles = userBuffer->getMaxDiffuseParticles();

		sDiffusePosLifeBuffer.initialize(cudaContextManager);
		sDiffusePosLifeBuffer.allocate(maxDiffuseParticles * sizeof(PxVec4));
	
		sPosBuffer.initialize(cudaContextManager);
		sPosBuffer.allocate(maxParticles * sizeof(PxVec4));
	}
}

void clearupParticleBuffers()
{
	sPosBuffer.release();
	sDiffusePosLifeBuffer.release();
}

void renderCallback()
{
	onBeforeRenderParticles();

	stepPhysics(true);

	Snippets::startRender(sCamera);

	PxScene* scene;
	PxGetPhysics().getScenes(&scene,1);
	PxU32 nbActors = scene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC | PxActorTypeFlag::eRIGID_STATIC);
	if(nbActors)
	{
		PxArray<PxRigidActor*> actors(nbActors);
		scene->getActors(PxActorTypeFlag::eRIGID_DYNAMIC | PxActorTypeFlag::eRIGID_STATIC, reinterpret_cast<PxActor**>(&actors[0]), nbActors);
		Snippets::renderActors(&actors[0], static_cast<PxU32>(actors.size()), true);
	}
	
	renderParticles();

	Snippets::showFPS();

	Snippets::finishRender();
}

void cleanup()
{
	delete sCamera;
	clearupParticleBuffers();
	cleanupPhysics(true);
}

void exitCallback()
{
}
}

void renderLoop()
{
	sCamera = new Snippets::Camera(PxVec3(15.0f, 10.0f, 15.0f), PxVec3(-0.6f,-0.2f,-0.6f));

	Snippets::setupDefault("PhysX Snippet PBFFluid", sCamera, keyPress, renderCallback, exitCallback);

	initPhysics(true);
	Snippets::initFPS();

	allocParticleBuffers();

	glutMainLoop();

	cleanup();
}
#endif
