/* ___________________________________________________________________________
 * Pyomo: Python Optimization Modeling Objects
 * Copyright (c) 2008-2022
 *  National Technology and Engineering Solutions of Sandia, LLC
 * Under the terms of Contract DE-NA0003525 with National Technology and
 * Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
 * rights in this software.
 * This software is distributed under the 3-clause BSD License.
 * ___________________________________________________________________________
*/
#include<fstream>
#include<unordered_map>
#include<vector>
#include<string>
#include<algorithm>

#include "funcadd.h"

unsigned int n_cspline_1d = 0;
std::unordered_map<std::string, unsigned int> idx_cspline_1d;
std::vector< std::vector<double> > joints_cspline_1d;
std::vector< std::vector<double> > a1_cspline_1d;
std::vector< std::vector<double> > a2_cspline_1d;
std::vector< std::vector<double> > a3_cspline_1d;
std::vector< std::vector<double> > a4_cspline_1d;
std::vector< unsigned int > n_seg_cspline_1d;


int read_parameters_cspline_1d(std::string file_path) {
   // Read data table
   try{
      // Parameters have been calculated already 
      return idx_cspline_1d.at(file_path);
   }
   catch(std::out_of_range const&){
      // Parameters haven't been read yet, so do that.
   }
   unsigned int i, idx;
   unsigned int n; // number of segments
   std::ifstream param_file; // parameter input file

   // Set index for cspline param file
   idx_cspline_1d[file_path] = n_cspline_1d;
   idx = n_cspline_1d;
   ++n_cspline_1d;

   // open the parameter file
   param_file.open(file_path);

   // get the number of segments
   param_file >> n;
   n_seg_cspline_1d.resize(n_cspline_1d);
   n_seg_cspline_1d[idx] = n;

   joints_cspline_1d.resize(n_cspline_1d);
   a1_cspline_1d.resize(n_cspline_1d);
   a2_cspline_1d.resize(n_cspline_1d);
   a3_cspline_1d.resize(n_cspline_1d);
   a4_cspline_1d.resize(n_cspline_1d);

   joints_cspline_1d[idx].resize(n + 1);
   a1_cspline_1d[idx].resize(n);
   a2_cspline_1d[idx].resize(n);
   a3_cspline_1d[idx].resize(n);
   a4_cspline_1d[idx].resize(n);

   // get the joints
   for(i=0; i < n + 1; ++i){
      param_file >> joints_cspline_1d[idx][i];
   }

   // get the a1 params
   for(i=0; i < n; ++i){
      param_file >> a1_cspline_1d[idx][i];
   }

   // get the a2 params
   for(i=0; i < n; ++i){
      param_file >> a2_cspline_1d[idx][i];
   }

   // get the a3 params
   for(i=0; i < n; ++i){
      param_file >> a3_cspline_1d[idx][i];
   }

   // get the a4 params
   for(i=0; i < n; ++i){
      param_file >> a4_cspline_1d[idx][i];
   }

   param_file.close();
   return idx;
}

extern real cspline_1d(arglist *al) {
   const char* data_file = al->sa[-(al->at[1]+1)];
   real x = al->ra[al->at[0]];
   unsigned int idx = read_parameters_cspline_1d(data_file);
   size_t seg;
   real a1, a2, a3, a4;

   //find segment index
   auto lit = std::lower_bound(joints_cspline_1d[idx].begin(), joints_cspline_1d[idx].end(), x);
   seg = lit - joints_cspline_1d[idx].begin();
   if(seg > 0) --seg;
   if(seg >= n_seg_cspline_1d[idx]) seg = n_seg_cspline_1d[idx] - 1;

   a1 = a1_cspline_1d[idx][seg];
   a2 = a2_cspline_1d[idx][seg];
   a3 = a3_cspline_1d[idx][seg];
   a4 = a4_cspline_1d[idx][seg];

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
      "cspline_1d", 
      (rfunc)cspline_1d,
      FUNCADD_REAL_VALUED|FUNCADD_STRING_ARGS, 
      2, 
      NULL
   );
}
