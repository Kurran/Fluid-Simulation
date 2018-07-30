#include "scalarfield.h"
#include <math.h>

#define _SCALAR_PI 3.14159

scalarfield::scalarfield()
{
	float h9;
	m_h = 0.2f;
	m_h2 = m_h * m_h;
	h9 = pow(m_h, 9);
	m_coefficient = 315.0 / (64.0 * _SCALAR_PI * h9);
}

float scalarfield::get_h()
{
	return m_h;
}

void scalarfield::set_h(float h)
{
	float h9 = pow(h,9);
	m_coefficient = 315.0f / (64.0f * _SCALAR_PI * h9);
	m_h2 = h*h;
}

float scalarfield::getfield(float x, float y, float z, float *pos, int npoints)
{
	float r2;
	float xd, yd, zd;
	float W = 0.0f;

	// Loop over every particle 

	for (int i = 0; i < npoints; i++)
	{
		xd = x - pos[4 * i];
		yd = y - pos[(4 * i) + 1];
		zd = z - pos[(4 * i) + 2];

		r2 = (xd*xd) + (yd*yd) + (zd*zd);

		if (r2 < m_h2)
		{
			W += m_coefficient * pow(m_h2 - r2, 3.0f);
		}

	}

	return W;
}

scalarfield::~scalarfield()
{
}
