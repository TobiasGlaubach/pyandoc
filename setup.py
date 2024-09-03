
import re

def get_property(prop, project):
    with open('src/' + project + '/__init__.py') as fp:
        result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop), fp.read())
    return result.group(1)

# print(get_property('__version__', "mke_client"))
from setuptools import setup; setup(version=get_property('__version__', "pydocmaker"))