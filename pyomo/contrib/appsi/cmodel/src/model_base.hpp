/**___________________________________________________________________________
 *
 * Pyomo: Python Optimization Modeling Objects
 * Copyright (c) 2008-2025
 * National Technology and Engineering Solutions of Sandia, LLC
 * Under the terms of Contract DE-NA0003525 with National Technology and
 * Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
 * rights in this software.
 * This software is distributed under the 3-clause BSD License.
 * ___________________________________________________________________________
**/

#ifndef MODEL_HEADER
#define MODEL_HEADER

#include "expression.hpp"

class Constraint;
class Objective;
class Model;

extern double inf;

class Objective {
public:
  Objective() = default;
  virtual ~Objective() = default;
  int sense = 0; // 0 means min; 1 means max
  std::string name;
};

class Constraint {
public:
  Constraint() = default;
  virtual ~Constraint() = default;
  std::shared_ptr<ExpressionBase> lb = std::make_shared<Constant>(-inf);
  std::shared_ptr<ExpressionBase> ub = std::make_shared<Constant>(inf);
  bool active = true;
  int index = -1;
  std::string name;
};

bool constraint_sorter(std::shared_ptr<Constraint> c1,
                       std::shared_ptr<Constraint> c2);

class Model {
public:
  Model();
  virtual ~Model() = default;
  std::set<std::shared_ptr<Constraint>, decltype(constraint_sorter) *>
      constraints;
  std::shared_ptr<Objective> objective;
  void add_constraint(std::shared_ptr<Constraint>);
  void remove_constraint(std::shared_ptr<Constraint>);
  int current_con_ndx = 0;
};

#endif
