from collections import namedtuple
import io
import json
import random
import textwrap
import time
import urllib
import re
import uuid
import os
import base64
import markdown
from typing import List


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


def convert(doc:List[dict]):
    tmp = doc.values() if isinstance(doc, dict) else doc
    return '\n\n'.join([html_docdc2html(dc) for dc in tmp])


class html_renderer:

    @staticmethod
    def vm_Text(**kwargs):
        label = kwargs.get('label', '')
        content = kwargs.get('content', kwargs.get('children'))
    
        if label:
            return f'<div style="min-width:100">{label}</div><div>{content}</div>'
        else:
            return f'<div>{content}</div>'
            
    @staticmethod
    def vm_Markdown(**kwargs):
        label = kwargs.get('label', '')
        content = kwargs.get('content', kwargs.get('children'))

        parts = []
        if label:
            parts += [
                f'<div style="min-width:100;">{label}</div>',
                '<hr/>'
            ]
        
        s = markdown.markdown(content)
        
        # s = f'<pre disabled=true style="width:90%; min-height:200px; overflow-x: scroll; overflow-y: none; margin:5px;display:block;font-family: Lucida Console, Courier New, monospace;font-size: 0.8em;">\n\n{content}\n\n</pre>'
        #s = f'<span style="display:block;" class="note">\n\n{content}\n\n</span>'
        parts += [s]

        return '\n\n'.join(parts)
    

    @staticmethod
    def vm_Verbatim(**kwargs):
        label = kwargs.get('caption', kwargs.get('label', ''))
        content = kwargs.get('content', kwargs.get('children'))

        j = content
        # nn = [len(s) for s in j.split('\n')]
        # n = len(nn)
        # w = max(nn)
        children = [
            f'<div style="min-width:100;">{label}</div>',
            # f'<textarea cols="{w}" rows="{n}" disabled=True>\n\n{j}\n\n</textarea>'
            f'<pre style="margin: 15px; margin-left: 25px; padding: 10px; border: 1px solid gray; border-radius: 3px;">{j}</pre>'
        ]
        return '\n\n'.join(children)

    @staticmethod
    def vm_Image(imageblob=None, children='', width=0.8, caption="", **kwargs):       
        
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
        
        # children = dcc.Upload(id=self.mkid('helper_uploadfile'), children=children, multiple=False, disable_click=True)

        return '\n\n'.join(children)

    @staticmethod
    def vm_Iterator( **kwargs):
        content = kwargs.get('content', kwargs.get('children'))
        return f'\n\n'.join([f'<div>{c}</div>' for c in content])


def html_docdc2html(content):

    if isinstance(content, str):
        return html_renderer.vm_Text(content=content)
    elif isinstance(content, dict) and content.get('typ', None) == 'iter' and isinstance(content.get('children', None), list):
        return html_renderer.vm_Iterator(content=[html_docdc2html(c) for c in content.get('children')])
    elif isinstance(content, list):
        return html_renderer.vm_Iterator(content=[html_docdc2html(c) for c in content])
    elif isinstance(content, dict) and content.get('typ', None) == 'image':
        return html_renderer.vm_Image(**content)
    elif isinstance(content, dict) and content.get('typ', None) == 'text':
        return html_renderer.vm_Text(**content) 
    elif isinstance(content, dict) and content.get('typ', None) == 'verbatim':
        return html_renderer.vm_Verbatim(**content) 
    elif isinstance(content, dict) and content.get('typ', None) == 'markdown':
        return html_renderer.vm_Markdown(**content) 
    else:
        raise TypeError(f'the element of type {type(content)}, could not be parsed.', content)
    



