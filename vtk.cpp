/*
 * vtk.cc
 *
 *  Created on: Mar 1, 2015
 *      Author: harpal
 */

#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <exception>
#include <string.h>
#include <iomanip>
#include "vtk.h"


vtk::vtk(const char* name)
{
	strncpy(_title , name, sizeof(_title));
	sprintf(_filename,"%s.vtk",name);
	// Open the file
	_fd_file.open(_filename);
	_data_points_added = 0;
	_data_points_to_add = 0;
}

vtk::~vtk()
{
	_fd_file.close();
}

void vtk::set_header()
{
	_fd_file << "# vtk DataFile Version 2.0" << std::endl ;
	_fd_file << _title << std::endl;
	_fd_file << "ASCII" << std::endl << std::endl;;
}

void vtk::add_cuboid(int x, int y, int z, int length, int breadth, int height)
{

	_fd_file << "DATASET POLYDATA" << std::endl;
	// Generate the set of eight points that define the cuboid
	_fd_file << "POINTS 8 int" << std::endl;

	for(int p=0; p<2; p++)
	{
		for(int q=0; q<2; q++)
		{
			for(int r=0; r<2; r++)
			{
				_fd_file << x + (r*length) << "  ";
				_fd_file << y + (q*breadth) << "  ";
				_fd_file << z + (p*height) << std::endl;
			}
		}
	}

	// Point Ordering

	// 0: (0,0,0)
	// 1: (1,0,0)
	// 2: (0,1,0)
	// 3: (1,1,0)
	// 4: (0,0,1)
	// 5: (1,0,1)
	// 6: (0,1,1)
	// 7: (1,1,1)

	// Generate the size rectangles required to generate a cuboid
	_fd_file << "POLYGONS 6 30" << std::endl;
	_fd_file << "4 0 2 6 4" << std::endl; // Rectangle at x = 0
	_fd_file << "4 1 3 7 5" << std::endl; // Rectangle at x = length
	_fd_file << "4 0 1 5 4" << std::endl; // Rectangle at y = 0
	_fd_file << "4 2 3 7 6" << std::endl; // Rectangle at y = breadth
	_fd_file << "4 0 1 3 2" << std::endl; // Rectangle at z = 0
	_fd_file << "4 4 5 7 6" << std::endl; // Rectangle at z = height

}


void vtk::add_points(int nPoints, float* data, size_t element_size)
{

	_fd_file << "DATASET POLYDATA" << std::endl;
	// Generate the set of eight points that define the cuboid
	_fd_file << "POINTS " << nPoints << " int" << std::endl;

	int q = 0;

	for(int p=0; p<nPoints; p++)
	{
		_fd_file << data[element_size*p] << "  ";
		_fd_file << data[element_size*p+1] << "  ";
		_fd_file << data[element_size*p+2] << "  ";
		_fd_file << std::endl;
	}

}


/*
 * @add_structured_data: Prints the contents of a 1D array as a 3D box.
 *
 * The 3D box has dimensions x_dim  x  y_dim  x  z_dim
 * The 1D array should store data as
 *
 * array[0]                x[0],y[0],z[0]
 * array[1]                x[1],y[0],z[0]
 * array[2]                x[2],y[0],z[0]
 *    .                        .
 *    .                        .
 * array[xdim-1]          x[xdim-1],y[0],z[0]
 * array[xdim]            x[0],y[1],z[0]
 * array[xdim+1]          x[1],y[1],z[0]
 * array[xdim+2]          x[2],y[1],z[0]
 *    .                        .
 *    .                        .
 * array[(2*xdim)-1]      x[dim-1],y[1],z[0]
 * array[2*xdim]          x[0],y[2],z[0]
 *    .                        .
 */

void vtk::add_structured_data(int x_dim, int y_dim, int z_dim,
		int x_offset, int y_offset, int z_offset, float* data)
{
	int vtk_points = x_dim * y_dim * z_dim;

	_fd_file << "DATASET STRUCTURED_GRID" << std::endl; // Rectangle at x = 0
	_fd_file << "DIMENSIONS" << "  " << x_dim << "  " << y_dim << "  " << z_dim << std::endl;
	_fd_file << "POINTS " << vtk_points << " float" << std::endl;
	_fd_file.width(8);
	_fd_file.precision(1);

	for(int z=0; z<z_dim; z++)
	{
		for(int y=0; y<y_dim; y++)
		{
			for(int x=0; x<x_dim; x++)
			{
				_fd_file << std::fixed << std::setw(4) << std::setprecision(1) << (float)(x + x_offset) << "  ";
				_fd_file << std::fixed << std::setw(4) << std::setprecision(1) << (float)(y + y_offset) << "  ";
				_fd_file << std::fixed << std::setw(4) << std::setprecision(1) << (float)(z + z_offset) << std::endl;
			}
		}
	}

	_fd_file << std::endl;
	_fd_file << "POINT_DATA   " << vtk_points << std::endl;
	_fd_file << "SCALARS scalars float" << std::endl;
	_fd_file << "LOOKUP_TABLE default" << std::endl;
	_fd_file.width(6);
	_fd_file.precision(3);

	float* p = data;
	for(int z=0; z<z_dim; z++)
	{
		for(int y=0; y<y_dim; y++)
		{
			for(int x=0; x<x_dim; x++)
			{
				_fd_file << std::fixed << std::setw(6) << std::setprecision(2)<< (*p) << "  ";
				p++;
			}
			_fd_file << std::endl;
		}
	}
}

void vtk::add_structured_data(int x_dim, int y_dim, int z_dim,
		int x_offset, int y_offset, int z_offset, unsigned char *data)
{
	int vtk_points = x_dim * y_dim * z_dim;


	_fd_file << "DATASET STRUCTURED_GRID" << std::endl; // Rectangle at x = 0
	_fd_file << "DIMENSIONS" << "  " << x_dim << "  " << y_dim << "  " << z_dim << std::endl;
	_fd_file << "POINTS " << vtk_points << " float" << std::endl;
	_fd_file.width(8);
	_fd_file.precision(1);

	for(int z=0; z<z_dim; z++)
	{
		for(int y=0; y<y_dim; y++)
		{
			for(int x=0; x<x_dim; x++)
			{
				_fd_file << std::fixed << std::setw(4) << std::setprecision(1) << (float)(x + x_offset) << "  ";
				_fd_file << std::fixed << std::setw(4) << std::setprecision(1) << (float)(y + y_offset) << "  ";
				_fd_file << std::fixed << std::setw(4) << std::setprecision(1) << (float)(z + z_offset) << std::endl;
			}
		}
	}

	_fd_file << std::endl;
	_fd_file << "POINT_DATA   " << vtk_points << std::endl;
	_fd_file << "SCALARS scalars float" << std::endl;
	_fd_file << "LOOKUP_TABLE default" << std::endl;
	_fd_file.width(6);
	_fd_file.precision(3);

	unsigned char *p = data;
	float  fdata;

	for(int z=0; z<z_dim; z++)
	{
		for(int y=0; y<y_dim; y++)
		{
			for(int x=0; x<x_dim; x++)
			{
				fdata = static_cast<float>(*p);
				_fd_file << std::fixed << std::setw(6) << std::setprecision(2)<< fdata << "  ";
				p++;
			}
			_fd_file << std::endl;
		}
	}
}
