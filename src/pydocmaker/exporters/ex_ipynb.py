
from collections import namedtuple
import io
import json
import random
import textwrap
import time
import traceback
import urllib
import re
import uuid
import os
import base64
import markdown
from typing import List


def txt2lines(txt):
    if isinstance(txt, str):
        txt = txt.split('\n')

    return [s + '\n' if not s.endswith('\n') else s for s in txt]

def make_raw(text):
    return {
   "cell_type": "raw",
   "metadata": {},
   "source": txt2lines(text)
}

def make_markdown(text):
    return {
   "cell_type": "markdown",
   "metadata": {},
   "source": txt2lines(text)
}

def squash_md(cells):
    return cells

    lines = []
    new_cells = []
    for cell in cells:
        if cell['cell_type'] == 'markdown':
            lines += ['\n', '\n'] + cell['source']
        elif lines: # case end of consecutive markdown
            new_cells.append(make_markdown(lines))
            new_cells.append(cell)
        else: # case no markdown
            new_cells.append(cell)

    return new_cells

def make_html(html_text):
    return {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": txt2lines(html_text),
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": []
  }


def make_doc(cells):
    return {
 "cells": cells,
"metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}



"""

 ██████  ██████  ███    ██ ██    ██ ███████ ██████  ████████ 
██      ██    ██ ████   ██ ██    ██ ██      ██   ██    ██    
██      ██    ██ ██ ██  ██ ██    ██ █████   ██████     ██    
██      ██    ██ ██  ██ ██  ██  ██  ██      ██   ██    ██    
 ██████  ██████  ██   ████   ████   ███████ ██   ██    ██    
                                                             
                                                                                                                                                                                                                                       
"""

# DEFAULT_IMAGE_PATH = os.path.join(parent_dir, 'ReqTracker', 'assets', 'mpifr.png')
# with open(DEFAULT_IMAGE_PATH, 'rb') as fp:
#     DEFAULT_IMAGE_BLOB = '' # base64.b64encode(fp.read()).decode('utf-8')
# DEFAULT_IMAGE_BLOB = ''

def mk_link(id_, label=None, pth='show', p0='uib', v='v1', **kwargs):
    return f'<a href="/{p0}/{v}/{pth}/{urllib.parse.quote_plus(id_)}" target="_self">{label if label else id_}</a>'

def mk_tpl(id_, label=None, pth='show', p0='uib', v='v1', **kwargs):
    return f"/{p0}/{v}/{pth}/{urllib.parse.quote_plus(id_)}", label if label else id_


def convert(doc:List[dict], as_dict=False):
    tmp = doc.values() if isinstance(doc, dict) else doc
    return ipynb_renderer().render(tmp, as_dict)
    


class ipynb_renderer:

    def __init__(self) -> None:
        self.cells = []

    def digest_Text(self,**kwargs):
        content = kwargs.get('content', kwargs.get('children'))
        self.cells += [make_raw(content)]
    
    def digest_Markdown(self,**kwargs):
        content = kwargs.get('content', kwargs.get('children'))
        self.cells += [make_markdown(content)]
        

    def digest_Verbatim(self,**kwargs):
        content = kwargs.get('content', kwargs.get('children'))
        self.cells += [make_raw(content)]

    def digest_Image(self,imageblob=None, children='', width=0.8, caption="", **kwargs):       
        
        if imageblob is None:
            imageblob = ''

        uid = (id(imageblob) + int(time.time()) + random.randint(1, 100))
        if not children:
            children = f'image_{uid}.png'

        s = imageblob.decode("utf-8") if isinstance(imageblob, bytes) else imageblob
        if not s.startswith('data:image'):
            s = 'data:image/png;base64,' + s

        children = [
            f'<div style="margin-top: 1.5em; width: 100%; text-align: center;"><span style="min-width:100;display: inline-block;"><b>image-name: </b>{children}</span></div>',
            f"<div style=\"width: 100%; text-align: center;\"><image src=\"{s}\", style=\"max-width:{int(width*100)}%;display: inline-block;\"></image></div>",
            f'<div style="width: 100%; text-align: center;"><span style="min-width:100;display: inline-block;"><b>caption: </b>{caption}</span></div>',
        ]
        
        html_text = '\n\n'.join(children)
        self.cells += [make_html(html_text)]

    
    def digest_Iterator(self, **kwargs):
        content = kwargs.get('content', kwargs.get('children'))
        for el in content:
            if el:
                self.digest(el)
        
    
    def handle_error(self, err, el) -> list:
        txt = 'ERROR WHILE HANDLING ELEMENT:\n{}\n\n'.format(el)
        if not isinstance(err, str):
            txt += '\n'.join(traceback.format_exception(err, limit=5)) + '\n'
        else:
            txt += err + '\n'
        txt = f'''<pre style="color:red;">\n{txt}\n</pre>'''

        self.cells += [make_html(txt)]

    def digest(self, content):
        try:
            if isinstance(content, str):
                self.digest_Text(content=content)
            elif isinstance(content, dict) and content.get('typ', None) == 'iter' and isinstance(content.get('children', None), list):
                self.digest_Iterator(content=[self.digest(c) for c in content.get('children')])
            elif isinstance(content, list):
                self.digest_Iterator(content=[self.digest(c) for c in content])
            elif isinstance(content, dict) and content.get('typ', None) == 'image':
                self.digest_Image(**content)
            elif isinstance(content, dict) and content.get('typ', None) == 'text':
                self.digest_Text(**content) 
            elif isinstance(content, dict) and content.get('typ', None) == 'verbatim':
                self.digest_Verbatim(**content) 
            elif isinstance(content, dict) and content.get('typ', None) == 'markdown':
                self.digest_Markdown(**content) 
            else:
                raise TypeError(f'the element of type {type(content)}, could not be parsed.', content)

        except Exception as err:
            return self.handle_error(err, content)
        
    def render(self, obj, as_dict=False):
        self.cells.clear()
        self.digest(obj)
        dc = make_doc(squash_md(self.cells))
        return dc if as_dict else json.dumps(dc, indent=2)
    