import atexit


def python_is_shutting_down():
    """Returns `True` if the interpreter is in the process of shutting down.

    This uses a function attribute to flag when the interpreter begins
    to shut down.  Note that we invert the sense of the flag: that way
    if the interpreter happens to have already cleared / released the
    flag and left it `None`, the correct answer is still returned.
    """
    return not python_is_shutting_down.isalive


python_is_shutting_down.isalive = [True]


@atexit.register
def _flag_shutting_down():
    python_is_shutting_down.isalive.clear()
