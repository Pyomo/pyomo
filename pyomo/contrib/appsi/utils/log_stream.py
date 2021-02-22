from io import TextIOBase


class LogStream(TextIOBase):
    def __init__(self, level, logger):
        self._level = level
        self._logger = logger

    def write(self, s: str) -> int:
        res = len(s)
        s = s.rstrip('\n')
        for line in s.split('\n'):
            self._logger.log(self._level, line)
        return res
