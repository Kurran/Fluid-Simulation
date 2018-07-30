#pragma once
class scalarfield
{
public:
	scalarfield();
	~scalarfield();

	float get_h(  );
	void set_h(float h);
	float getfield(float x, float y, float z, float *pos, int npoints);

private:

	float m_h;
	float m_h2;
	float m_coefficient;

};

