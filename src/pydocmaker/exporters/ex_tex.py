
import argparse
import base64
import os
import re
from pathlib import Path
import json
import shutil
import traceback
import sys

import tempfile
import shutil
from io import BytesIO

from dataclasses import dataclass
import zipfile

from typing import List
import markdown
try:
    import pydocmaker.sub.mdx_latex as mdx_latex
except Exception as err:
    from ..sub import mdx_latex

md = markdown.Markdown()
latex_mdx = mdx_latex.LaTeXExtension()
latex_mdx.extendMarkdown(md)






def convert(doc:List[dict], with_attachments=True, files_to_upload=None):

    files_to_upload = {} if not files_to_upload else files_to_upload

    formatter = ElementFormatter()
    s = formatter.format(doc)
    
    text = '\n'.join(s) if isinstance(s, list) else s
    if with_attachments:
        return text, formatter.attachments
    else:
        return text
    




###########################################################################################
"""

███████  ██████  ██████  ███    ███  █████  ████████ 
██      ██    ██ ██   ██ ████  ████ ██   ██    ██    
█████   ██    ██ ██████  ██ ████ ██ ███████    ██    
██      ██    ██ ██   ██ ██  ██  ██ ██   ██    ██    
██       ██████  ██   ██ ██      ██ ██   ██    ██    
                                                     
"""
###########################################################################################


class ElementFormatter:

    def __init__(self, make_blue=False) -> None:
        self.attachments = {}
        self.make_blue = make_blue


    def handle_error(self, err, el):
        txt = 'ERROR WHILE HANDLING ELEMENT:\n{}\n\n'.format(el)
        if not isinstance(err, str):
            txt += '\n'.join(traceback.format_exception(err, limit=5)) + '\n'
        else:
            txt += err + '\n'
        txt = r"""
\begin{verbatim}

<REPLACEME:VERBTEXT>

\end{verbatim}""".replace('<REPLACEME:VERBTEXT>', txt)
        txt = f'{{\\color{{red}}{txt}}}'

        return txt

    def digest_markdown(self, children='', **kwargs) -> str:
        tex = md.convert(children).lstrip('<root>').rstrip('</root>')
        return tex

    
    def digest_image(self, children='', width=0.8, caption='', imageblob='', **kwargs) -> str:

        if not isinstance(width, str):
            width = 'width={}\\textwidth'.format(width)

        file_name = os.path.basename(children)
        relpath = os.path.join('inp', file_name)
        path = './inp/' + file_name

        if imageblob:
            if isinstance(imageblob, str):
                if ';base64, ' in imageblob:
                    imageblob = imageblob.replace(';base64, ', ';base64,')

                imageblob = imageblob.encode("utf8")

            data = imageblob.split(b";base64,")[-1]
            self.attachments[relpath] = base64.decodebytes(data)

        txt = fr'\includegraphics[{width}]{{{path}}}'

        if caption:
            txt += '\n' + fr'\caption{{{caption}}}'

        txt = r"\begin{figure}[h!]" + '\n' + r"\centering" '\n' + txt + '\n' + r"\end{figure}"
        return txt



    def digest_verbatim(self, children='', **kwargs) -> str:
        txt = self.digest(children)
        template = r"""\begin{tabular}{|p{16cm}|}
\hline
\begin{tiny}\begin{verbatim}
<REPLACEME:VERBTEXT>
\end{verbatim}\end{tiny}
\\
\hline
\end{tabular}\par"""
        txt = txt.strip('\n')
        parts = []

        while len(txt) > 2000:
            parts.append(template.replace('<REPLACEME:VERBTEXT>', txt[:2000]))
            txt = txt[2000:]
        parts.append(template.replace('<REPLACEME:VERBTEXT>', txt))

        txt = '\n\n'.join(parts)
        # if caption:
        #     caption = fr'\caption{{{caption}}}'

        # txt = txt.replace('<REPLACEME:CAPTION>', caption)

        return txt


    def digest_iterator(self, el) -> str:
        if isinstance(el, dict) and el.get('typ', '') == 'iter' and isinstance(el.get('children', None), list):
            el = el['children']
        return '\n\n'.join([f'% Iterator Element {i}\n' + self.digest(e) for i, e in enumerate(el)])
    
    def digest_str(self, el):
        return el
        
    def digest_text(self, children:str, **kwargs):
        return children
    
    def digest(self, el, make_blue=False):
        blue = lambda s: f'{{\\color{{blue}}{s}}}'
        try:
            
            if not el:
                return ''
            elif isinstance(el, str):
                ret = self.digest_str(el)
            elif isinstance(el, dict) and 'typ' in el and el['typ'] == 'iter':
                ret = self.digest_iterator(el)
            elif isinstance(el, list) and el:
                ret = self.digest_iterator(el)
            elif isinstance(el, dict) and 'typ' in el and el['typ'] == 'image':
                ret = self.digest_image(**el)
            elif isinstance(el, dict) and 'typ' in el and el['typ'] == 'text':
                ret = self.digest_text(**el)
            elif isinstance(el, dict) and 'typ' in el and el['typ'] == 'verbatim':
                ret = self.digest_verbatim(**el)
            elif isinstance(el, dict) and 'typ' in el and el['typ'] == 'markdown':
                ret = self.digest_markdown(**el)
            else:
                return self.handle_error(f'the element of typ {type(el)}, could not be parsed.', el)
            
            return blue(ret) if make_blue else ret
        
        except Exception as err:
            return self.handle_error(err, el)


    def format(self, doc:list) -> str:
        return '\n\n'.join([self.digest(e, make_blue=self.make_blue) for e in doc])











