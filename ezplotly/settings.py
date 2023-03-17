from enum import Enum

import plotly
from IPython import get_ipython

SUPPRESS_PLOTS: bool = False


class EnvironmentSettings(Enum):
    """
    Enum for environment settings.
    """

    JUPYTER = "Jupyter"
    TERMINAL = "Terminal"
    UNKNOWN = "Unknown"


def get_environment() -> EnvironmentSettings:
    """
    Gets the environment settings of plotly.
    :return: the environment that the code is running in.
    """
    shell = get_ipython().__class__.__name__
    if shell == "ZMQInteractiveShell":
        return EnvironmentSettings.JUPYTER
    elif shell == "TerminalInteractiveShell":
        return EnvironmentSettings.TERMINAL
    else:
        return EnvironmentSettings.UNKNOWN


# initialize plotly for jupyter notebook environment
if get_environment() == EnvironmentSettings.JUPYTER:
    plotly.offline.init_notebook_mode(connected=True)
