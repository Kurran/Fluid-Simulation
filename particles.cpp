/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
	Particle system example with collisions using uniform grid

	CUDA 2.1 SDK release 12/2008
	- removed atomic grid method, some optimization, added demo mode.

	CUDA 2.2 release 3/2009
	- replaced sort function with latest radix sort, now disables v-sync.
	- added support for automated testing and comparison to a reference value.
	*/

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (WIN32)
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif
#include "vtk.h"


// CUDA runtime
#include <cuda_runtime.h>

//#include "scalarfield.h"

//#include "marching_cubes.h"

// CUDA utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda_gl.h> // includes cuda_gl_interop.h// includes cuda_gl_interop.h

// Includes
#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

#include "particleSystemGpu.h"
#include "particleSystemCpu.h"
#include "marchingCubes.h"
#include "render_particles.h"
#include "paramgl.h"

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f

#define GRID_SIZE       64
#define NUM_PARTICLES   24576

const uint width = 640, height = 480;

// view params
int ox, oy;
int buttonState = 0;
float camera_trans[] = { 0, 0, -3 };
float camera_rot[] = { 0, 0, 0 };
float camera_trans_lag[] = { 0, 0, -3 };
float camera_rot_lag[] = { 0, 0, 0 };
const float inertia = 0.1f;
ParticleRenderer::DisplayMode displayMode = ParticleRenderer::PARTICLE_SPHERES;

#define REFRESH_DELAY     10 //ms

int mode = 0;
bool displayEnabled = true;
bool bPause = false;
bool displaySliders = false;
bool wireframe = false;
bool demoMode = false;
int idleCounter = 0;
int demoCounter = 0;
const int idleDelay = 2000;


unsigned int  frame_skip_rate = 1;

enum { M_VIEW = 0, M_MOVE };

uint numParticles = 0;
uint3 gridSize;
int numIterations = 0; // run until exit

GLint  gl_Shader;

// simulation parameters
float timestep = 0.5f;
float damping = 1.0f;
float gravity = 0.0003f;
int iterations = 1;
int ballr = 10;

float collideSpring = 0.5f;
float collideDamping = 0.02f;
float collideShear = 0.1f;
float collideAttraction = 0.0f;

ParticleSystem *psystem = 0;
MarchingCubes  *mcubes = 0;

// fps
static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;

ParticleRenderer *renderer = 0;

float modelView[16];

ParamListGL *params;

float3 mc_rotate = make_float3(0.0, 0.0, 0.0);
float3 mc_translate= make_float3(0.0, 0.0, -3.0);


// Auto-Verification Code
const int frameCheckNumber = 4;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
char        *g_refFile = NULL;

const char *sSDKsample = "CUDA Particles Simulation";

extern "C" void cudaInit(int argc, char **argv);
extern "C" void cudaGLInit(int argc, char **argv);
extern "C" void copyArrayFromDevice(void *host, const void *device, unsigned int vbo, int size);


bool step_frame = false;

int filenum = 0;

// shader for displaying floating-point texture
static const char *shader_code =
    "!!ARBfp1.0\n"
    "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
    "END";

GLuint compileASMShader(GLenum program_type, const char *code)
{
    GLuint program_id;
    glGenProgramsARB(1, &program_id);
    glBindProgramARB(program_type, program_id);
    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei) strlen(code), (GLubyte *) code);

    GLint error_pos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

    if (error_pos != -1)
    {
        const GLubyte *error_string;
        error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
        fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos, error_string);
        return 0;
    }

    return program_id;
}


void initParticleSystem(int numParticles, uint3 gridSize, bool bUseOpenGL)
{
	if (bUseOpenGL)
		printf("Using opengl.\n");
	else
		printf("NOT Using OpenGL !!!\n");

	psystem = new ParticleSystemGpu(numParticles, gridSize, bUseOpenGL);

	fprintf(stderr, "Resetting  ParticleSystem!\n");
	psystem->reset(ParticleSystem::CONFIG_GRID);

	if (bUseOpenGL)
	{
		fprintf(stderr, "Creating particle renderer!\n");
		renderer = new ParticleRenderer;
		renderer->setParticleRadius(psystem->getParticleRadius());
		renderer->setColorBuffer(psystem->getColorBuffer());
	}

	fprintf(stderr, "Creating Timer!\n");
	sdkCreateTimer(&timer);
}


void initMarchingCubes(uint thrice_log_gridSize, bool bUseOpenGL)
{
	fprintf(stderr, "Creating marching cubes!\n");
	mcubes = new MarchingCubes();
	fprintf(stderr, "Initialising marching cubes!\n");
	mcubes->init(thrice_log_gridSize, bUseOpenGL);
	fprintf(stderr, "Setting marching cubes field!\n");
	mcubes->setVolume(psystem->getField());
}


void cleanup()
{
	sdkDeleteTimer(&timer);

	if (psystem)
	{
		delete psystem;
	}

	if (mcubes)
	{
		delete mcubes;
	}
	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();
	return;
}


bool initGL(int *argc, char **argv)
{
    // Create GL context
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA Marching Cubes");

    // initialize necessary OpenGL extensions
    glewInit();

    if (! glewIsSupported("GL_VERSION_2_0 "
                         ))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
    glEnable(GL_DEPTH_TEST);

    // good old-fashioned fixed function lighting
    float black[]    = { 0.0f, 0.0f, 0.0f, 1.0f };
    float white[]    = { 1.0f, 1.0f, 1.0f, 1.0f };
    float light_blue[] = { 0.0f, 0.5f, 1.0f, 0.2f };

    float ambient[]  = { 0.1f, 0.1f, 0.1f, 1.0f };
    float diffuse[]  = { 0.5f, 0.5f, 0.5f, 1.0f };
    float lightPos[] = { 0.0f, 0.0f, 1.0f, 0.0f };

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black);

    glLightfv(GL_LIGHT0, GL_AMBIENT, light_blue);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_blue);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_blue);
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, black);

    glEnable(GL_LIGHT0);
    glEnable(GL_NORMALIZE);

    // load shader program
    gl_Shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);

    glutReportErrors();

    return true;
}


void runBenchmark(int iterations, char *exec_path)
{
	printf("Run %u particles simulation for %d iterations...\n\n", numParticles, iterations);
	cudaDeviceSynchronize();
	sdkStartTimer(&timer);

	for (int i = 0; i < iterations; ++i)
	{
		psystem->update(timestep, false);
	}

	cudaDeviceSynchronize();
	sdkStopTimer(&timer);
	float fAvgSeconds = ((float)1.0e-3 * (float)sdkGetTimerValue(&timer) / (float)iterations);

	printf("particles, Throughput = %.4f KParticles/s, Time = %.5f s, Size = %u particles, NumDevsUsed = %u, Workgroup = %u\n",
		(1.0e-3 * numParticles) / fAvgSeconds, fAvgSeconds, numParticles, 1, 0);

	if (g_refFile)
	{
		printf("\nChecking result...\n\n");
		float *hPos = (float *)malloc(sizeof(float) * 4 * psystem->getNumParticles());
		copyArrayFromDevice(hPos, psystem->getCudaPosVBO(),
			0, sizeof(float) * 4 * psystem->getNumParticles());

		sdkDumpBin((void *)hPos, sizeof(float) * 4 * psystem->getNumParticles(), "particles.bin");

		if (!sdkCompareBin2BinFloat("particles.bin", g_refFile, sizeof(float) * 4 * psystem->getNumParticles(),
			MAX_EPSILON_ERROR, THRESHOLD, exec_path))
		{
			g_TotalErrors++;
		}
	}
}


void computeFPS()
{
	frameCount++;
	fpsCount++;

	if (fpsCount == fpsLimit)
	{
		char fps[256];
		float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		sprintf(fps, "CUDA Particles (%d particles): %3.1f fps  Isovalue %4.2f  : skip %d",
				numParticles, ifps, mcubes->m_isoValue, frame_skip_rate);

		glutSetWindowTitle(fps);
		fpsCount = 0;

		fpsLimit = (int)MAX(ifps, 1.f);
		sdkResetTimer(&timer);
	}
}


void animation()
{
        mcubes->m_isoValue += mcubes->m_dIsoValue;

        if (mcubes->m_isoValue < 0.1f)
        {
        	mcubes->m_isoValue = 0.1f;
        	mcubes->m_dIsoValue *= -1.0f;
        }
        else if (mcubes->m_isoValue > 0.9f)
        {
        	mcubes->m_isoValue = 0.9f;
        	mcubes->m_dIsoValue *= -1.0f;
        }
}


void timerEvent(int value)
{
    animation();
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}


void display()
{

	static unsigned int frame_count = 0;

	sdkStartTimer(&timer);



	// update the simulation
	if (!bPause)
	{
		psystem->setIterations(iterations);
		psystem->setDamping(damping);
		psystem->setGravity(-gravity);
		psystem->setCollideSpring(collideSpring);
		psystem->setCollideDamping(collideDamping);
		psystem->setCollideShear(collideShear);
		psystem->setCollideAttraction(collideAttraction);

		/*
		 * Update particle positions and do collision detection
		 * The resultant field is computed here too
		 */
		//if(step_frame)
		//{
		//  step_frame = false;
		  psystem->update(timestep, frame_count%frame_skip_rate == 0 );
		//}


		// TODO:	Currently psystem->update() updates the particle positions and updates the scalar field.
		//			Pull the scalar field out of particleSystem code and put into its own class

		// Use the scalar field to compute the isosurface

		if(frame_count%frame_skip_rate == 0 ) mcubes->computeIsosurface();



		if (renderer)
		{
			renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumParticles());
		}
	}


    // Common display code path

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(mc_translate.x, mc_translate.y, mc_translate.z);

    glRotatef(90.0, 1.0, 0.0, 0.0);

    glRotatef(mc_rotate.x, 1.0, 0.0, 0.0);
    glRotatef(mc_rotate.y, 0.0, 1.0, 0.0);
    glRotatef(mc_rotate.z, 0.0, 0.0, 1.0);

    glPolygonMode(GL_FRONT_AND_BACK, wireframe? GL_LINE : GL_FILL);

    glEnable(GL_LIGHTING);


    // cube
    glColor3f(1.0, 1.0, 1.0);
    glutWireCube(2.0);

    // render
    glPushMatrix();
    glRotatef(180.0, 0.0, 1.0, 0.0);
    glRotatef(90.0, 1.0, 0.0, 0.0);
    mcubes->renderIsosurface();
    glPopMatrix();

    glDisable(GL_LIGHTING);


	sdkStopTimer(&timer);

	glutSwapBuffers();
	glutReportErrors();

	computeFPS();
	frame_count++;
}


inline float frand()
{
	return rand() / (float)RAND_MAX;
}


void addSphere()
{
	// inject a sphere of particles
	float pr = psystem->getParticleRadius();
	float tr = pr + (pr*2.0f)*ballr;
	float pos[4], vel[4];
	pos[0] = -1.0f + tr + frand()*(2.0f - tr*2.0f);
	pos[1] = 1.0f - tr;
	pos[2] = -1.0f + tr + frand()*(2.0f - tr*2.0f);
	pos[3] = 0.0f;
	vel[0] = vel[1] = vel[2] = vel[3] = 0.0f;
	psystem->addSphere(0, pos, vel, ballr, pr*2.0f);
}


void reshape(int w, int h)
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (float)w / (float)h, 0.1, 10.0);

	glMatrixMode(GL_MODELVIEW);
	glViewport(0, 0, w, h);

	if (renderer)
	{
		renderer->setWindowSize(w, h);
		renderer->setFOV(60.0);
	}
}


void mouse(int button, int state, int x, int y)
{
	int mods;

	if (state == GLUT_DOWN)
	{
		buttonState |= 1 << button;
	}
	else if (state == GLUT_UP)
	{
		buttonState = 0;
	}

	//mods = glutGetModifiers();

	//if (mods & GLUT_ACTIVE_SHIFT)
	//{
	//	buttonState = 2;
	//}
	//else if (mods & GLUT_ACTIVE_CTRL)
	//{
	//	buttonState = 3;
	//}

	ox = x;
	oy = y;

	//demoMode = false;
	//idleCounter = 0;

	//if (displaySliders)
	//{
	//	if (params->Mouse(x, y, button, state))
	//	{
	//		glutPostRedisplay();
	//		return;
	//	}
	//}

	//glutPostRedisplay();
}


// transform vector by matrix
void xform(float *v, float *r, GLfloat *m)
{
	r[0] = v[0] * m[0] + v[1] * m[4] + v[2] * m[8] + m[12];
	r[1] = v[0] * m[1] + v[1] * m[5] + v[2] * m[9] + m[13];
	r[2] = v[0] * m[2] + v[1] * m[6] + v[2] * m[10] + m[14];
}


// transform vector by transpose of matrix
void ixform(float *v, float *r, GLfloat *m)
{
	r[0] = v[0] * m[0] + v[1] * m[1] + v[2] * m[2];
	r[1] = v[0] * m[4] + v[1] * m[5] + v[2] * m[6];
	r[2] = v[0] * m[8] + v[1] * m[9] + v[2] * m[10];
}


void ixformPoint(float *v, float *r, GLfloat *m)
{
	float x[4];
	x[0] = v[0] - m[12];
	x[1] = v[1] - m[13];
	x[2] = v[2] - m[14];
	x[3] = 1.0f;
	ixform(x, r, m);
}


void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - ox);
	dy = (float)(y - oy);

	//if (displaySliders)
	//{
	//	if (params->Motion(x, y))
	//	{
	//		ox = x;
	//		oy = y;
	//		glutPostRedisplay();
	//		return;
	//	}
	//}

    if (buttonState==1)
    {
        mc_rotate.x += dy * 0.2f;
        mc_rotate.y += dx * 0.2f;
    }
    else if (buttonState==2)
    {
    	mc_translate.x += dx * 0.01f;
    	mc_translate.y -= dy * 0.01f;
    }
    else if (buttonState==3)
    {
    	mc_translate.z += dy * 0.01f;
    }

    ox = x;
    oy = y;

    /*
	switch (mode)
	{
	case M_VIEW:
		if (buttonState == 3)
		{
			// left+middle = zoom
			camera_trans[2] += (dy / 100.0f) * 0.5f * fabs(camera_trans[2]);
		}
		else if (buttonState & 2)
		{
			// middle = translate
			camera_trans[0] += dx / 100.0f;
			camera_trans[1] -= dy / 100.0f;
		}
		else if (buttonState & 1)
		{
			// left = rotate
			camera_rot[0] += dy / 5.0f;
			camera_rot[1] += dx / 5.0f;
		}

		break;

	case M_MOVE:
	{
		float translateSpeed = 0.003f;
		float3 p = psystem->getColliderPos();

		if (buttonState == 1)
		{
			float v[3], r[3];
			v[0] = dx*translateSpeed;
			v[1] = -dy*translateSpeed;
			v[2] = 0.0f;
			ixform(v, r, modelView);
			p.x += r[0];
			p.y += r[1];
			p.z += r[2];
		}
		else if (buttonState == 2)
		{
			float v[3], r[3];
			v[0] = 0.0f;
			v[1] = 0.0f;
			v[2] = dy*translateSpeed;
			ixform(v, r, modelView);
			p.x += r[0];
			p.y += r[1];
			p.z += r[2];
		}

		psystem->setColliderPos(p);
	}
	break;
	}

	ox = x;
	oy = y;

	demoMode = false;
	idleCounter = 0;
	*/

	glutPostRedisplay();
}


void key(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key)
	{
	case ' ':
		bPause = !bPause;
		break;

	case 13:


		break;

	case '\033':
	case 'q':
#if defined(__APPLE__) || defined(MACOSX)
		exit(EXIT_SUCCESS);
#else
		glutDestroyWindow(glutGetWindow());
		return;
#endif
	case 'v':
		mode = M_VIEW;
		break;

	case 'm':
		mode = M_MOVE;
		break;

	case 'b':
		step_frame = true;
		break;

	case 'p':
		displayMode = (ParticleRenderer::DisplayMode)
			((displayMode + 1) % ParticleRenderer::PARTICLE_NUM_MODES);
		break;

	case 'd':
		psystem->dumpGrid();
		break;

	case 'u':
		psystem->dumpParticles(0, 100);
		break;

	case 'r':
		displayEnabled = !displayEnabled;
		break;

	case '1':
		psystem->reset(ParticleSystem::CONFIG_GRID);
		break;

	case '2':
		psystem->reset(ParticleSystem::CONFIG_RANDOM);
		break;

	case '3':
		addSphere();
		break;

	case '4':
	{
		// shoot ball from camera
		float pr = psystem->getParticleRadius();
		float vel[4], velw[4], pos[4], posw[4];
		vel[0] = 0.0f;
		vel[1] = 0.0f;
		vel[2] = -0.05f;
		vel[3] = 0.0f;
		ixform(vel, velw, modelView);

		pos[0] = 0.0f;
		pos[1] = 0.0f;
		pos[2] = -2.5f;
		pos[3] = 1.0;
		ixformPoint(pos, posw, modelView);
		posw[3] = 0.0f;

		psystem->addSphere(0, posw, velw, ballr, pr*2.0f);
		break;
	}
	
	case '5':
	{
		printf("number of particles %d", psystem->getNumParticles());
		
	}

	/*
	 * This is a debug function to capture the current state of the
	 * field and record it in a vtk file to be viewed later
	 *
	 */

	case '6':
	{
		vtk vtkop("field");
		vtkop.set_header();
		int xd,yd,zd;

		xd = yd = zd = GRID_SIZE;

		fprintf(stderr, "Saving field data to vtk file...\n");

		vtkop.add_structured_data(xd,yd,zd,0,0,0,psystem->dumpField() );

		fprintf(stderr, "Saved.\n");
		break;
	}

	case '7':
	{
		char particles_filename[56];
		static int particles_dump_count = 0;
		sprintf(particles_filename,"particles_%d", particles_dump_count++);
		vtk vtkop(particles_filename);
		vtkop.set_header();

		fprintf(stderr, "Saving particle data to vtk file...\n");

		vtkop.add_points(psystem->getNumParticles(),
				psystem->getArray(ParticleSystem::POSITION),
				4);

		fprintf(stderr, "Saved.\n");
		break;
	}


	case 'w':
		wireframe = !wireframe;
		break;

	case 'h':
		displaySliders = !displaySliders;
		break;

    case '=':
         mcubes->m_isoValue += 0.5f;
         break;

     case '-':
    	 mcubes->m_isoValue -= 0.5f;
         break;

     case '+':
    	 mcubes->m_isoValue += 0.5f;
         break;

     case '_':
    	 mcubes->m_isoValue -= 0.5f;
         break;

     case 'g':
    	 if(frame_skip_rate<32)
    		 frame_skip_rate++;
         break;

     case 'f':
    	 if(frame_skip_rate>1)
    		 frame_skip_rate--;
         break;

	}

	demoMode = false;
	idleCounter = 0;
	glutPostRedisplay();
}


void special(int k, int x, int y)
{
	if (displaySliders)
	{
		params->Special(k, x, y);
	}

	demoMode = false;
	idleCounter = 0;
}


void idle(void)
{
	/*
	if ((idleCounter++ > idleDelay) && (demoMode == false))
	{
		//demoMode = true;
		//printf("Entering demo mode\n");
	}

	if (demoMode)
	{
		camera_rot[1] += 0.1f;

		if (demoCounter++ > 1000)
		{
			ballr = 10 + (rand() % 10);
			addSphere();
			demoCounter = 0;
		}
	}
	*/
	//animation();
	glutPostRedisplay();
}


void initParams()
{
	if (g_refFile)
	{
		timestep = 0.0f;
		damping = 0.0f;
		gravity = 0.0f;
		ballr = 1;
		collideSpring = 0.0f;
		collideDamping = 0.0f;
		collideShear = 0.0f;
		collideAttraction = 0.0f;
	}
	else
	{
		// create a new parameter list
		params = new ParamListGL("misc");
		params->AddParam(new Param<float>("time step", timestep, 0.0f, 1.0f, 0.01f, &timestep));
		params->AddParam(new Param<float>("damping", damping, 0.0f, 1.0f, 0.001f, &damping));
		params->AddParam(new Param<float>("gravity", gravity, 0.0f, 0.001f, 0.0001f, &gravity));
		params->AddParam(new Param<int>("ball radius", ballr, 1, 20, 1, &ballr));

		params->AddParam(new Param<float>("collide spring", collideSpring, 0.0f, 1.0f, 0.001f, &collideSpring));
		params->AddParam(new Param<float>("collide damping", collideDamping, 0.0f, 0.1f, 0.001f, &collideDamping));
		params->AddParam(new Param<float>("collide shear", collideShear, 0.0f, 0.1f, 0.001f, &collideShear));
		params->AddParam(new Param<float>("collide attract", collideAttraction, 0.0f, 0.1f, 0.001f, &collideAttraction));
	}
}


void mainMenu(int i)
{
	key((unsigned char)i, 0, 0);
}


void initMenus()
{
	glutCreateMenu(mainMenu);
	glutAddMenuEntry("Reset block [1]", '1');
	glutAddMenuEntry("Reset random [2]", '2');
	glutAddMenuEntry("Add sphere [3]", '3');
	glutAddMenuEntry("View mode [v]", 'v');
	glutAddMenuEntry("Move cursor mode [m]", 'm');
	glutAddMenuEntry("Toggle point rendering [p]", 'p');
	glutAddMenuEntry("Toggle animation [ ]", ' ');
	glutAddMenuEntry("Step animation [ret]", 13);
	glutAddMenuEntry("Toggle sliders [h]", 'h');
	glutAddMenuEntry("Quit (esc)", '\033');
	glutAttachMenu(GLUT_RIGHT_BUTTON);
}


int main(int argc, char **argv)
{
#if defined(__linux__)
	setenv("DISPLAY", ":0", 0);
#endif

	printf("%s Starting...\n\n", sSDKsample);

	printf("NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n\n");

	numParticles = NUM_PARTICLES;
	uint gridDim = GRID_SIZE;
	numIterations = 0;

	if (argc > 1)
	{
		if (checkCmdLineFlag(argc, (const char **)argv, "n"))
		{
			numParticles = getCmdLineArgumentInt(argc, (const char **)argv, "n");
		}

		if (checkCmdLineFlag(argc, (const char **)argv, "grid"))
		{
			gridDim = getCmdLineArgumentInt(argc, (const char **)argv, "grid");
		}

		if (checkCmdLineFlag(argc, (const char **)argv, "file"))
		{
			getCmdLineArgumentString(argc, (const char **)argv, "file", &g_refFile);
			fpsLimit = frameCheckNumber;
			numIterations = 1;
		}
	}

	gridSize.x = gridSize.y = gridSize.z = gridDim;
	printf("grid: %d x %d x %d = %d cells\n", gridSize.x, gridSize.y, gridSize.z, gridSize.x*gridSize.y*gridSize.z);
	printf("particles: %d\n", numParticles);

	bool benchmark = checkCmdLineFlag(argc, (const char **)argv, "benchmark") != 0;

	if (checkCmdLineFlag(argc, (const char **)argv, "i"))
	{
		numIterations = getCmdLineArgumentInt(argc, (const char **)argv, "i");
	}

	if (g_refFile)
	{
		cudaInit(argc, argv);
	}
	else
	{
		if (checkCmdLineFlag(argc, (const char **)argv, "device"))
		{
			printf("[%s]\n", argv[0]);
			printf("   Does not explicitly support -device=n in OpenGL mode\n");
			printf("   To use -device=n, the sample must be running w/o OpenGL\n\n");
			printf(" > %s -device=n -file=<*.bin>\n", argv[0]);
			printf("exiting...\n");
			exit(EXIT_SUCCESS);
		}

		initGL(&argc, argv);
		cudaGLInit(argc, argv);
	}

    //initParticleSystem(numParticles, gridSize, g_refFile==NULL);

	bool using_openGL = false;
	fprintf(stderr, "Particle system is %s OpenGL\n", using_openGL? "using":"not using");
	initParticleSystem(numParticles, gridSize, using_openGL);
    initParams();


	if (!g_refFile)
	{
		initMenus();
	}

	if (benchmark || g_refFile)
	{
		if (numIterations <= 0)
		{
			numIterations = 300;
		}

		runBenchmark(numIterations, argv[0]);
	}
	else
	{
		glutDisplayFunc(display);
		glutReshapeFunc(reshape);
		glutMouseFunc(mouse);
		glutMotionFunc(motion);
		glutKeyboardFunc(key);
		glutSpecialFunc(special);
		glutIdleFunc(idle);
		glutCloseFunc(cleanup);
		//glutTimerFunc(REFRESH_DELAY, timerEvent,0);
		initMarchingCubes( 3 * (int)(floor(log2(double(gridDim)))), true);

		glutMainLoop();

	}

	if (psystem)
	{
		delete psystem;
	}
	if (mcubes)
	{
		delete mcubes;
	}


	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();
	exit(g_TotalErrors > 0 ? EXIT_FAILURE : EXIT_SUCCESS);
}

