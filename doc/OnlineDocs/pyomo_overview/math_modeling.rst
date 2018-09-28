Mathematical Modeling
---------------------

This section provides an introduction to Pyomo: Python Optimization
Modeling Objects.  A more complete description is contained in the
[PyomoBookII]_ book. Pyomo supports the formulation and analysis of
mathematical models for complex optimization applications.  This
capability is commonly associated with commerically available algebraic
modeling languages (AMLs) such as [AMPL]_, [AIMMS]_, and [GAMS]_.
Pyomo's modeling objects are embedded within Python, a full-featured,
high-level programming language that contains a rich set of supporting
libraries.

Modeling is a fundamental process in many aspects of scientific
research, engineering and business.  Modeling involves the formulation
of a simplified representation of a system or real-world object.  Thus,
modeling tools like Pyomo can be used in a variety of ways:

- *Explain phenomena* that arise in a system,

- *Make predictions* about future states of a system,

- *Assess key factors* that influence phenomena in a system,

- *Identify extreme states* in a system, that might represent worst-case
  scenarios or minimal cost plans, and

- *Analyze trade-offs* to support human decision makers.

Mathematical models represent system knowledge with a formalized
language.  The following mathematical concepts are central to modern
modeling activities:

Variables
*********
    
    Variables represent unknown or changing parts of a model (e.g.,
    whether or not to make a decision, or the characteristic of a system
    outcome). The values taken by the variables are often referred to as
    a *solution* and are usually an output of the optimization process.

Parameters
**********
    
    Parameters represents the data that must be supplied to perform the
    optimization. In fact, in some settings the word *data* is used in
    place of the word *parameters*.

Relations
*********
    
    These are equations, inequalities or other mathematical
    relationships that define how different parts of a model are
    connected to each other.

Goals
*****
    
    These are functions that reflect goals and objectives for the system
    being modeled.

The widespread availability of computing resources has made the
numerical analysis of mathematical models a commonplace activity.
Without a modeling language, the process of setting up input files,
executing a solver and extracting the final results from the solver
output is tedious and error-prone.  This difficulty is compounded in
complex, large-scale real-world applications which are difficult to
debug when errors occur.  Additionally, there are many different formats
used by optimization software packages, and few formats are recognized
by many optimizers.  Thus the application of multiple optimization
solvers to analyze a model introduces additional complexities.

Pyomo is an AML that extends Python to include objects for mathematical
modeling. [PyomoBookI]_, [PyomoBookII]_, and [PyomoJournal]_ compare
Pyomo with other AMLs.  Although many good AMLs have been developed for
optimization models, the following are motivating factors for the
development of Pyomo:

Open Source
***********

    Pyomo is developed within Pyomo's open source project to promote
    transparency of the modeling framework and encourage community
    development of Pyomo capabilities.

Customizable Capability
***********************
 
    Pyomo supports a customizable capability through the extensive use
    of plug-ins to modularize software components.

Solver Integration
******************
  
    Pyomo models can be optimized with solvers that are written either
    in Python or in compiled, low-level languages.

Programming Language
********************
  
    Pyomo leverages a high-level programming language, which has several
    advantages over custom AMLs: a very robust language, extensive
    documentation, a rich set of standard libraries, support for modern
    programming features like classes and functions, and portability to
    many platforms.
