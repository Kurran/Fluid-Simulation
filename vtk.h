/*
 * vtk.h
 *
 *  Created on: Mar 1, 2015
 *      Author: harpal
 */

#ifndef _VTK_H_
#define _VTK_H_

#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <exception>
#include <string.h>

//using namespace std;

class vtk
{
protected:

	char _filename[56];
	char _title[56];
	std::ofstream  _fd_file;

	int _data_points_added;
	int _data_points_to_add;

public:

	void set_header();

	void add_structured_data(int x_dim, int y_dim, int z_dim, float* data);

	void add_structured_data(
			int x_dim, int y_dim, int z_dim,
			int x_offset, int y_offset, int z_offset,
			float* data);

	void add_structured_data(
			int x_dim, int y_dim, int z_dim,
			int x_offset, int y_offset, int z_offset,
			unsigned char *data);

	void add_cuboid(int x, int y, int z, int length, int breadth, int height);

	void add_points(int n_points, float* data, size_t element_size);

	vtk(const char* name);

	~vtk();
};



#endif /* _VTK_H_ */



