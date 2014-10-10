__all__ = ['IPreCheckHook', 'IPostCheckHook']

from coopr.core.plugin import *

class IPreCheckHook(Interface):

    def precheck(self, runner, script, info):
        pass


class IPostCheckHook(Interface):

    def postcheck(self, runner, script, info):
        pass
