from coopr.pyomo.plugins.check.checker import IterativeTreeChecker

class PrintASTNodes(IterativeTreeChecker):

    def __init__(self):
        self.disable()

    def check(self, runner, script, info):
        if 'lineno' in dir(info):
            self.problem(str(info), lineno = info.lineno)
        else:
            self.problem(str(info))
