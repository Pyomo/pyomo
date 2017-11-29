.. |p| raw:: html

   <p />

Design Details
==============

Expressions inherit from ExpressionBase

Expression classes typically represent binary operations.

Expressions are immutable

Special expression nodes
* Named expressions
* Linear expressions
* ViewSum expressions
* Mutable expressions

Expressions classes are treated as potentially variable except
for derived not-potentially-variable classes.

There are three types of expressions:

* constant expressions - do not contain numeric constants and immutable parameters
* mutable expressions - contain mutable parameters but no variables
* potentially variable expresions - contain variables


    m.p = Param(default=10, mutable=False)
    m.q = Param(default=10, mutable=True)
    m.x = var()
    m.y = var(initialize=1)
    m.y.fixed = True

                            m.p     m.q     m.x     m.y
    constant                T       F       F       F
    potentially_variable    F       F       T       T
    npv                     T       T       F       F
    fixed                   T       T       F       T
