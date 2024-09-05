__version__ = '1.3.4'

from pydocmaker.core import DocBuilder, construct, constr, buildingblocks, print_to_pdf
from pydocmaker.util import upload_report_to_redmine


def get_schema():
    return {k: getattr(constr, k)() for k in buildingblocks}
        