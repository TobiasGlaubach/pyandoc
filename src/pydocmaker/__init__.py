__version__ = '1.1.2'

from pydocmaker.core.schema import FlowDoc, SectionedDoc, construct, constr, buildingblocks

from pydocmaker.exporters.ex_docx import convert as to_docx
from pydocmaker.exporters.ex_html import convert as to_html
from pydocmaker.exporters.ex_redmine import convert as to_redmine
from pydocmaker.exporters.ex_redmine import convert as to_textile
from pydocmaker.exporters.ex_tex import convert as to_tex

def get_schema():
    return {k: getattr(constr, k)() for k in buildingblocks}
        