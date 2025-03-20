/* ___________________________________________________________________________
 * Pyomo: Python Optimization Modeling Objects
 * Copyright (c) 2008-2025
 *  National Technology and Engineering Solutions of Sandia, LLC
 * Under the terms of Contract DE-NA0003525 with National Technology and
 * Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
 * rights in this software.
 * This software is distributed under the 3-clause BSD License.
 * ___________________________________________________________________________
*/
#include<iostream>
#include<fstream>
#include<sstream>
#include<unordered_map>
#include<vector>
#include<string>
#include<algorithm>

#include "funcadd.h"

// Don't want to have the overhead of reading parameter files
// each time, so keep params as long as library remains loaded

// Number of loaded curves
unsigned int n_cspline = 0;

// Parameter file path to unsigned in index map
std::unordered_map<std::string, unsigned int> idx_cspline;

// Knots for each curve
std::vector< std::vector<double> > knots_cspline;

// Cubic segment parameters for each curve
std::vector< std::vector<double> > a1_cspline;
std::vector< std::vector<double> > a2_cspline;
std::vector< std::vector<double> > a3_cspline;
std::vector< std::vector<double> > a4_cspline;

// Number of segments in each curve
std::vector< unsigned int > n_seg_cspline;

// The data file format is a text file with one parameter per line
//   Line 1: Number of parameters
//   nsegs + 1 lines for knots
//   nsegs lines for a1
//   nsegs lines for a2
//   nsegs lines for a3
//   nsegs lines for a4 

// Function to read parameter file if not read and store params
static int read_parameters_cspline(std::string file_path) {
   try{ //if read already return curve index and done.
      return idx_cspline.at(file_path);
   }
   catch(std::out_of_range const&){} // Not read so read it.
   unsigned int i, idx, n; //loop counter, curve index, and number of segs
   
   // Set index for cspline param file and increment curve count
   idx_cspline[file_path] = n_cspline;
   idx = n_cspline;
   ++n_cspline;

   // open the parameter file for input
   // Assume if there is a newline in the file_path, that it is actually
   // a string with the file contents.
   std::ifstream file_stream;
   std::istringstream string_stream(file_path);

   bool is_fname = false;
   if(file_path.find('\n') == std::string::npos){
      file_stream.open(file_path);
      is_fname = true;
   }

   std::istream& param_file = is_fname ? 
      static_cast<std::istream&>(file_stream)
      : static_cast<std::istream&>(string_stream);

   // get the number of segments and size to vectors
   param_file >> n;
   n_seg_cspline.resize(n_cspline);
   n_seg_cspline[idx] = n;
   knots_cspline.resize(n_cspline);
   a1_cspline.resize(n_cspline);
   a2_cspline.resize(n_cspline);
   a3_cspline.resize(n_cspline);
   a4_cspline.resize(n_cspline);
   knots_cspline[idx].resize(n + 1);
   a1_cspline[idx].resize(n);
   a2_cspline[idx].resize(n);
   a3_cspline[idx].resize(n);
   a4_cspline[idx].resize(n);

   // get the knots
   for(i=0; i < n + 1; ++i){
      param_file >> knots_cspline[idx][i];
   }

   // get the a1 params
   for(i=0; i < n; ++i){
      param_file >> a1_cspline[idx][i];
   }

   // get the a2 params
   for(i=0; i < n; ++i){
      param_file >> a2_cspline[idx][i];
   }

   // get the a3 params
   for(i=0; i < n; ++i){
      param_file >> a3_cspline[idx][i];
   }

   // get the a4 params
   for(i=0; i < n; ++i){
      param_file >> a4_cspline[idx][i];
   }

   file_stream.close();

   // Returns curve index
   return idx;
}

// Calculate the cspline 1D function value and derivatives
extern real cspline(arglist *al) {
   // Get the data file path argument
   const char* data_file = al->sa[-(al->at[1]+1)];
   // Get the function input
   real x = al->ra[al->at[0]];
   // Get parameters, read file if needed
   unsigned int idx = read_parameters_cspline(data_file);
   size_t seg; // segment index
   real a1, a2, a3, a4; // segment parameters

   // Find segment index
   auto lit = std::lower_bound(knots_cspline[idx].begin(), knots_cspline[idx].end(), x);
   seg = lit - knots_cspline[idx].begin();

   // If off the curve on the low side use first seg (0) otherwise
   // subtract 1 to go from knot index to segment index (i.e. seg is
   // initially how many knots are below, so 0 is less than first knot,
   // 1 is above first knot..., so > 0 we need to decrement to get the 
   // segment index)
   if(seg > 0) --seg;
   // If off the curve on the high side, use last segment
   if(seg >= n_seg_cspline[idx]) seg = n_seg_cspline[idx] - 1;

   // Get the parameters for the correct segment 
   a1 = a1_cspline[idx][seg];
   a2 = a2_cspline[idx][seg];
   a3 = a3_cspline[idx][seg];
   a4 = a4_cspline[idx][seg];

   // Compute the first derivative, if requested.
   if (al->derivs!=NULL) {
      al->derivs[0] = a2 + 2*a3*x + 3*a4*x*x;
      // Compute the second derivative, if requested.
      if (al->hes!=NULL) {
         al->hes[0] = 2*a3 + 6*a4*x;
      }
   }

   return a1 + a2*x + a3*x*x + a4*x*x*x;
}

// Register external functions defined in this library with the ASL
void funcadd(AmplExports *ae){
   // Arguments for addfunc (this is not fully detailed; see funcadd.h)
   // 1) Name of function in AMPL
   // 2) Function pointer to C function
   // 3) see FUNCADD_TYPE enum in funcadd.h
   // 4) Number of arguments
   //    >=0 indicates a fixed number of arguments
   //    < 0 indicates a variable length list (requiring at least -(n+1)
   //        arguments)
   // 5) Void pointer to function info
   addfunc(
      "cspline", 
      (rfunc)cspline,
      FUNCADD_REAL_VALUED|FUNCADD_STRING_ARGS, 
      2, 
      NULL
   );
}
