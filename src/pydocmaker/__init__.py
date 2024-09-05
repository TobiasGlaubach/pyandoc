__version__ = '1.3.0'

from pydocmaker.core import DocBuilder, construct, constr, buildingblocks


def get_schema():
    return {k: getattr(constr, k)() for k in buildingblocks}
        