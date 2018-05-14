"""
David L. Woodruff and Mingye Yang, Spring 2018
Code snippets for BuildAction.rst in testable form
"""

# @BuildAction_example
>>> model.BuildBpts = BuildAction(model.J, rule=bpts_build)
# @BuildAction_example

# @Function_valid_declaration
>>> def bpts_build(model, j):
# @Function_valid_declaration
