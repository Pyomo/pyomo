\section{Functions to Support Modeling \label{sec:components:fcts}}

Pyomo includes a variety of functions that support the construction
and evaluation of model expressions.   Functions that are used to construct nonlinear expressions are        described
in Chapter~\ref{chap:nonlinear} (see Table~\ref{table:nonlinear:functions} on page \pageref{table:nonlinear: functions}).
Table~\ref{table:components3:utility} summarizes the utility functions
that support the concise expression of modeling concepts.  These functions
are described further in this section.

\begin{table}[tbp]

\begin{tabular}{p{1.8in} p{2.7in}} \hline
\code{display} \index{display function@\code{display} function} & Display the properties of models and model components\\
\code{dot\_product} \index{dot\_product@\code{dot\_product} function} & Compute a generalized dot product\\
\code{sequence} \index{sequence@\code{sequence} function} & Compute a sequence of integers\\
\code{sum_product} \index{sum_product@\code{sum_product} function} & Compute a generalized dot product\\
\code{value} \index{value@\code{value} function} & Compute the value of a model component\\
\code{xsequence} \index{xsequence@\code{xsequence} function} & Compute a sequence of integers\\ \hline
\end{tabular}

\caption{\label{table:components3:utility} Pyomo utility functions that support the construction and         evaluation of model expressions.}
\end{table}

\subsection{Generalized Dot Products}

The \code{sum_product}\index{sum_product@\code{sum_product} function|textbf} function is a utility function that   computes a generalized
dot product;  the \code{dot\_product}\index{dot\_product@\code{dot\_product} function|textbf} is a synonym   for this function.
This function creates an expression that represents the sum of elements of one or
more indexed components.  We use the following components in our examples:
\listing{examples/utility/summation.py}{components}{7}{14}
In the simplest case, \code{summation} creates an expression that
represents the sum of the elements of an indexed component.  For example,
\listing{examples/utility/summation.py}{sum1}{18}{18}
represents the sum $\sum_{i=1}^3 x_i$.  This function provides a convenient shorthand for defining           expressions in objectives and constraints.  For example, the
following constraint uses the Python \code{sum}\index{python!sum@\code{sum} function} function and a Python  generator expression to define the constraint body:
\listing{examples/utility/summation.py}{c1}{22}{23}
This constraint can be rewritten in a more concise format with the \code{summation} function:
\listing{examples/utility/summation.py}{c2}{27}{27}

More generally, the \code{summation} function can be used to create
an expression for dot products between two arrays of parameters and
variables in addition to generalized dot products and sums of subexpressions.
For example, the following objective computes a simple dot product
between \code{a} and \code{x}:
\listing{examples/utility/summation.py}{o1}{31}{31}

The \code{denom} option is used to create terms that are fractions.  This option specifies one or more       components that are divided into each term of the sum:
\listing{examples/utility/summation.py}{o2}{35}{35}
The objective in this expression represents the sum $\sum_{i=1}^3 x_i/y_i$.
Similarly, two or more terms can be included in the denominator with the following
syntax:
\listing{examples/utility/summation.py}{o3}{39}{40}

The previous examples have constructed sums from components with the
same index set.  When components have different index sets, the index
set is inferred from the component arguments.  If one or more components
are specified for the numerator, then the last component specified defines the index
for the sum.  For example, consider the following:
\listing{examples/utility/summation.py}{o4}{44}{45}
This sum represents the polynomial $x_1 z_1/a_1 + x_3 z_3/a_3$.  The index for component \code{z} defines    the index for this sum, since it occurs last.  Similarly, if no
numerator components are specified, then the last component specified with the \code{denom} option defines   the index for the sum.  For example, consider the following:
\listing{examples/utility/summation.py}{o5}{49}{50}
This sum represents the polynomial $\frac{1}{x_1 z_1} + \frac{1}{x_3 z_3}$.

Finally, the \code{index} option allows the explicit specification of
an index set.  For example:
\listing{examples/utility/summation.py}{o6}{54}{55}
This sum represents $x_1 y_1 + x_3 y_3$.

\subsection{Generating Sequences}

The function \code{sequence([start,] stop[, step])}\index{sequence@\code{sequence} function|textbf} returns  a list that
is an arithmetic progression of integers.  With a single argument,
\code{sequence} returns a sequence that starts with \code{1}.  Thus,
\code{sequence(i)} returns \code{[1, 2, ..., i]}.  With two arguments,
\code{sequence(i, j)} returns \code{[i, i+1, i+2, ..., j]}.  The third
argument, when given, specifies the increment (or decrement if negative).

Note that \code{sequence} is simply a wrapper around the Python
\code{range} function.  The main difference is that the lists are
shifted by one.  The lists returned by \code{range} start at \code{0},
and the lists returned by \code{sequence} start at \code{1}.  Thus,
\code{sequence} has a functionality that is more familiar to mathematical
modelers.

The function \code{xsequence}\index{xsequence@\code{xsequence} function|textbf} returns a Python generator   for the list
that is created by \code{sequence}.  A generator constructs the numbers
in the sequence on demand.  For looping, this is slightly faster and
more memory efficient.


\subsection{Helper Functions}

Pyomo includes several helper functions that aid in the interrogation of model objects.  Pyomo models and    model components support a \code{display}\index{display function@\code{display} function|textbf} method,     which summarizes the model.  The \code{display} function is a helper function that can be called directly to generate this summary.  For example, consider the following simple
model:
\listing{examples/utility/display.py}{Model}{4}{10}
The command \code{display(model)} displays the entire model:
\listing{examples/utility/display.txt}{display1}{2}{18}
and the \code{display(model.x)} command displays the setup of the \code{x} variable:
\listing{examples/utility/display.txt}{display2}{21}{23}

Various components of a Pyomo model have a \textit{value} that represents
The \code{value} function provides a wrapper for accessing and computing the value of Pyomo components.  For example, \code{value(model.x)} returns the value of \code{x} variable:
\listing{examples/utility/display.txt}{value2}{29}{29}
and \code{value(model.o)} compute the value of the expression for the \code{o} objective:
\listing{examples/utility/display.txt}{value1}{26}{26}

