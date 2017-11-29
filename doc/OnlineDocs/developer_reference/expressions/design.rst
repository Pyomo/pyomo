.. |p| raw:: html

   <p />

Expression Classes
==================

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

