from dataclasses import dataclass, field, is_dataclass
from collections import UserDict, UserList
import json, io
import os
import re
import time
import zipfile
import requests
import base64
import copy



from ..exporters.ex_docx import convert as _to_docx
from ..exporters.ex_html import convert as _to_html
from ..exporters.ex_redmine import convert as _to_textile
from ..exporters.ex_tex import convert as _to_tex


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter




class constr():
    """This is the basic schema for the main building blocks for a document"""

    @staticmethod
    def markdown(children=''):
        return {
            'typ': 'markdown',
            'children': children
        }
    
    @staticmethod
    def text(children=''):
        return {
            'typ': 'text',
            'children': children
        }
    
    @staticmethod
    def verbatim(children=''):
        return {
            'typ': 'verbatim',
            'children': children
        }
    
    @staticmethod
    def iter(children:list=None):
        return {
            'typ': 'iter',
            'children': [] if children is None else children,
        }
    
    @staticmethod
    def image(imageblob='', caption='', children='', width=0.8):
        return {
            'typ': 'image',
            'children': children,
            'imageblob': imageblob.decode("utf-8") if isinstance(imageblob, bytes) else imageblob,
            'caption': caption,
            'width': width,
        }
    

    @staticmethod
    def image_from_link(url, caption='', children='', width=0.8):

        assert url, 'need to give an URL!'

        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        
        mime_type = response.headers.get("Content-Type")
        assert mime_type.startswith('image'), f'the downloaded content does not seem to be of any image type! {mime_type=}'
        
        if not children:
            children = url.split('/')[-1]
            if children.startswith('File:'):
                children = children[len('File:'):]
        
        children = children.strip()
        
        if not caption and children:
            caption = children

        children = re.sub(r'[^a-zA-Z0-9._-]', '', children)

        if not '.' in children:
            children += '.' + mime_type.split('/')[-1]

        imageblob = base64.b64encode(response.content).decode('utf-8')
        return constr.image(imageblob=imageblob, children=children, caption=caption, width=width)
    


    @staticmethod
    def image_from_file(path, children='', caption='', width=0.8):

        assert path, 'need to give a path!'

        if hasattr(path, 'read'):
            bts = path.read()
        else:
            with open(path, 'rb') as fp:
                bts = fp.read()
        
        assert bts and isinstance(bts, bytes), f'the loaded content needs to be of type bytes but was {bts=}'
        
        if not children:
            children = os.path.basename(path)
        
        if not caption and children:
            caption = children

        imageblob = base64.b64encode(bts).decode('utf-8')
        return constr.image(imageblob=imageblob, children=children, caption=caption, width=width)
        

    def image_from_fig(caption='', width=0.8, name=None, fig=None, **kwargs):
        """convert a matplotlib figure (or the current figure) to a document image dict to later add to a document

        Args:
            caption (str, optional): the caption to give to the image. Defaults to ''.
            width (float, optional): The width for the image to have in the document. Defaults to 0.8.
            name (str, optional): A specific name/id to give to the image (will be auto generated if None). Defaults to None.
            fig (matplotlib figure, optional): the figure which to upload (or the current figure if None). Defaults to None.

        Returns:
            dict
        """
        if not 'plt' in locals():
            import matplotlib.pyplot as plt

        if fig:
            plt.figure(fig)

        with io.BytesIO() as buf:
            plt.savefig(buf, format='png', **kwargs)
            buf.seek(0)   

            img = base64.b64encode(buf.read()).decode('utf-8')
        
        if name is None:
            id_ = str(id(img))[-2:]
            name = f'figure_{int(time.time())}_{id_}.png'

        return constr.image(imageblob = 'data:image/png;base64,' + img, children=name, caption=caption, width=width)


    @staticmethod
    def image_from_obj(im, caption = '', width=0.8, name=None):
        """make a image type dict from given image of type matrix, filelike or PIL image

        Args:
            im (np.array): the image as NxMx
            caption (str, optional): the caption to give to the image. Defaults to ''.
            width (float, optional): The width for the image to have in the document. Defaults to 0.8.
            name (str, optional): A specific name/id to give to the image (will be auto generated if None). Defaults to None.

        Returns:
            dict with the results
        """
        if not 'np' in locals():
            import numpy as np
        if not 'Image' in locals():
            from PIL import Image

        # 2D matrix as lists --> make nummpy array
        if isinstance(im, list) and im and im[0] and isinstance(im[0], list):
            im = np.array(im)

        # numpy array --> make PIL image
        if hasattr(im, 'shape') and len(im.shape) == 2:
            im = Image.fromarray(im)
        
        # PIL image --> make filelike
        if hasattr(im, 'save'):
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            buf.seek(0)   
            im = buf

        # filepath --> make filelike
        if isinstance(im, str) and os.path.exists(im):
            if not name:
                name = os.path.basename(im)            
            im = open(im, 'rb')

        # file like --> make bytes
        if hasattr(im, 'read'):
            im.seek(0)   
            im = im.read()
        
        # bytes --> make b64 string
        if isinstance(im, bytes):
            im = base64.b64encode(im).decode('utf-8')

        if name is None:
            id_ = str(id(im))[-2:]
            name = f'image_{int(time.time())}_{id_}.png'

        return constr.image(imageblob = 'data:image/png;base64,' + im, children=name, caption=caption, width=width)

buildingblocks = 'text markdown image verbatim iter'.split()

class FlowDoc(UserList):
    """an unordered collection of document parts to make a document (can be used like a list)"""

    def add(self, part:dict=None):
        """append a new document part to the end of this document

        Args:
            part (dict): the part to add. see the constr class for all possible parts.
        """
        assert part, 'need to give an element_to_add!'
        
        if isinstance(part, str):
            part = constr.text(part)

        assert hasattr(constr, part.get('typ', None)), 'the part to add is of unknown type!'
        self.append(part)
            
    def add_kw(self, typ, children=None, **kwargs):
        """add a document part to this document with a given typ

        Args:
            typ (str, optional): one of the allowed document part types. Either 'markdown', 'verbatim', 'text', 'iter' or 'image'.
            children (str or list): the "children" for this element. Either text directly (as string) or a list of other parts
            kwargs: the kwargs for such a document part
        """

        assert typ, 'need to give a content type!'
        self.add(construct(typ, children=children, **kwargs))

    def dump(self):
        """dump this document to a basic list of dicts for document parts

        Returns:
            list: the individual parts of the document
        """
        return [copy.deepcopy(v) for v in self]
    
    def _ret(self, m, path_or_stream):
        if isinstance(path_or_stream, str):
            with open(path_or_stream, "wb") as f:
                f.write(m)
            return True
        elif hasattr(path_or_stream, 'write'):
            path_or_stream.write(m)
            return True
        else:
            return m
        
    def to_json(self, path_or_stream=None) -> str:
        """
        Converts the current object to a JSON file.

        Args:
            path_or_stream (str or io.IOBase, optional): The path to save the file to, or a file-like object to write the data to. If not provided, the data will be returned as string.

        Returns:
            str: The JSON data as string, or True if the data was saved successfully to a file or stream.
        """
        return self._ret(json.dumps(self.dump(), indent=4), path_or_stream)

    def to_docx(self, path_or_stream=None) -> bytes:
        """
        Converts the current object to a DOCX file.

        Args:
            path_or_stream (str or io.IOBase, optional): The path to save the file to, or a file-like object to write the data to. If not provided, the data will be returned as string.

        Returns:
            str: The data as bytes, or True if the data was saved successfully to a file or stream.
        """
        return self._ret(_to_docx(self.dump()), path_or_stream)        
    
    def to_html(self, path_or_stream=None) -> str:
        """
        Converts the current object to a HTML file.

        Args:
            path_or_stream (str or io.IOBase, optional): The path to save the file to, or a file-like object to write the data to. If not provided, the data will be returned as string.

        Returns:
            str: The data as string, or True if the data was saved successfully to a file or stream.
        """
        return self._ret(_to_html(self.dump()), path_or_stream)
    
    def to_tex(self, path_or_stream=None):
        """
        Converts the current object to a TEX file (and attachments).

        Args:
            path_or_stream (str or io.IOBase, optional): The path to save the file to, or a file-like object to write the data to. If not provided, the data will be returned as string.

        Returns:
            case saving to file or stream:
                True if the data was saved successfully to a file or stream.
            case returning:
                str: The tex file as string
                dict: The additional input files needed for LateX (bytes) as values and their relative pathes (str) as keys
        """
        
        tex, files = _to_tex(self.dump(), with_attachments=True)
        with io.BytesIO() as in_memory_zip:
            with zipfile.ZipFile(in_memory_zip, 'w') as zipf:
                zipf.writestr('doc.json', self.to_json())
                zipf.writestr('main.tex', tex)
                for path in files:
                    zipf.writestr(path, files[path])
            in_memory_zip.seek(0)
            m = in_memory_zip.getvalue()

        if isinstance(path_or_stream, str):
            with open(path_or_stream, "wb") as f:
                f.write(m)
            return True
        elif hasattr(path_or_stream, 'write'):
            path_or_stream.write(m)
            return True
        else:
            return tex, files
    
    def to_textile(self, path_or_stream=None):
        """
        Converts the current object to a TEXTILE file (and attachments). 
        If path_or_stream is given it will zip all contents and write it to the stream or file path given.
        If not it will return a tuple with textile (str), files (dict[str, bytes])

        Args:
            path_or_stream (str or io.IOBase, optional): The path to save the file to, or a file-like object to write the data to. If not provided, the data will be returned as string.
        """
        
        textile, files = _to_textile(self.dump(), with_attachments=True, aformat_redmine=False)
        with io.BytesIO() as in_memory_zip:
            with zipfile.ZipFile(in_memory_zip, 'w') as zipf:
                # zipf.writestr('doc.json', self.to_json())
                zipf.writestr('main.textile', textile)
                for path in files:
                    zipf.writestr(path, files[path])
            in_memory_zip.seek(0)
            m = in_memory_zip.getvalue()

        if isinstance(path_or_stream, str):
            with open(path_or_stream, "wb") as f:
                f.write(m)
            return True
        elif hasattr(path_or_stream, 'write'):
            path_or_stream.write(m)
            return True
        else:
            return textile, files
        
    def to_redmine(self):
        """
        Converts the current object to a Redmine Textile like text (and attachments) and returns them as tuple
        """
        return _to_textile(self.dump(), with_attachments=True, aformat_redmine=True)
    

    def show(self):
        if is_notebook():
            from IPython.display import display, HTML
            return display(HTML(self.to_html()))
        else:
            return print(self.to_json())

class SectionedDoc(UserDict):
    """a sectioned collection of document parts to make a document (can be used like a dict)"""

    def add_section(self, caption, children:list=None):
        assert caption, 'need to give a caption!'
        assert not caption in self, f'section with {caption=} already exists in Document'
        assert children is None or isinstance(children, list), 'children must be of type list or None'
        self[caption] = FlowDoc() if children is None else children

    def add(self, section_caption:str, part:dict):
        """append a new document part to the end of this document

        Args:
            part (dict): the part to add. see the constr class for all possible parts.
        """
        if not section_caption and len(self):
            section_caption = list(self.keys())[-1]
        
        assert section_caption, 'need to give a section_caption to add to!'
        assert part, 'need to give an element_to_add!'

        if isinstance(part, str):
            part = constr.text(part)

        assert hasattr(constr, part.get('typ', None)), 'the element to add is of unknown type!'
        if not section_caption in self:
            self.add_section(caption=section_caption)
        self[section_caption].add(part)
            
    def add_kw(self, section_caption, typ, children=None, **kwargs):
        """add a document part to this document with a given typ

        Args:
            section_caption (str): the key/name/caption to add this part to
            typ (str): one of the allowed document part types. Either 'markdown', 'verbatim', 'text', 'iter' or 'image'.
            children (str or list): the "children" for this element. Either text directly (as string) or a list of other parts
            kwargs: the kwargs for such a document part
        """
        
        assert typ, 'need to give a content type!'
        self.add(section_caption, construct(typ, children=children, **kwargs))

    def to_flow_doc(self):
        doc = FlowDoc()
        for section_caption, section_parts in self.items():
            doc.add(constr.markdown(f'## {section_caption}'))
            for part in section_parts:
                doc.add(part)
        return doc
    
    def dump(self):
        return self.to_flow_doc().dump()


    def to_json(self, path_or_stream=None) -> str:
        """
        Converts the current object to a JSON file.

        Args:
            path_or_stream (str or io.IOBase, optional): The path to save the file to, or a file-like object to write the data to. If not provided, the data will be returned as string.

        Returns:
            str: The JSON data as string, or True if the data was saved successfully to a file or stream.
        """
        return self.to_flow_doc().to_json(path_or_stream)

    def to_docx(self, path_or_stream=None) -> bytes:
        """
        Converts the current object to a DOCX file.

        Args:
            path_or_stream (str or io.IOBase, optional): The path to save the file to, or a file-like object to write the data to. If not provided, the data will be returned as string.

        Returns:
            str: The data as bytes, or True if the data was saved successfully to a file or stream.
        """
        return self.to_flow_doc().to_docx(path_or_stream)
    
    def to_html(self, path_or_stream=None) -> str:
        """
        Converts the current object to a HTML file.

        Args:
            path_or_stream (str or io.IOBase, optional): The path to save the file to, or a file-like object to write the data to. If not provided, the data will be returned as string.

        Returns:
            str: The data as string, or True if the data was saved successfully to a file or stream.
        """
        return self.to_flow_doc().to_html(path_or_stream)
    
    def to_tex(self, path_or_stream=None):
        """
        Converts the current object to a TEX file (and attachments).

        Args:
            path_or_stream (str or io.IOBase, optional): The path to save the file to, or a file-like object to write the data to. If not provided, the data will be returned as string.

        Returns:
            case saving to file or stream:
                True if the data was saved successfully to a file or stream.
            case returning:
                str: The tex file as string
                dict: The additional input files needed for LateX (bytes) as values and their relative pathes (str) as keys
        """
        return self.to_flow_doc().to_tex(path_or_stream)
    
    def to_textile(self, path_or_stream=None):
        """
        Converts the current object to a TEXTILE file (and attachments). 
        If path_or_stream is given it will zip all contents and write it to the stream or file path given.
        If not it will return a tuple with textile (str), files (dict[str, bytes])

        Args:
            path_or_stream (str or io.IOBase, optional): The path to save the file to, or a file-like object to write the data to. If not provided, the data will be returned as string.
        """
        return self.to_flow_doc().to_textile(path_or_stream)
        
    
    def to_redmine(self):
        """
        Converts the current object to a Redmine Textile like text (and attachments) and returns them as tuple
        """
        return self.to_flow_doc().to_redmine()
    


    def show(self):
        if is_notebook():
            from IPython.display import HTML
            return HTML(self.to_html())
        else:
            return self.to_json()

# def _serialize(v):
#     if isinstance(v, (str, float)):
#         return v
#     elif isinstance(v, list):
#         return [dump(vv) for vv in v]
#     elif isinstance(v, dict):
#         return v
#     else:
#         TypeError(f'{type(v)=} is of unknown type only dataclass, str, list, and dict is allowed!')


def _construct(v):

    if isinstance(v, str):
        return v
    elif isinstance(v, list):
        return [_construct(vv) for vv in v]
    elif isinstance(v, dict):
        return construct(**v)
    else:
        TypeError(f'{type(v)=} is of unknown type only dataclass, str, list, and dict is allowed!')

def construct(type:str, **kwargs):
    assert isinstance(type, str)
    if not kwargs and not hasattr(constr, type):
        return type
    elif hasattr(constr, type):
        children = kwargs.get('children')
        if children:
            kwargs['children'] = _construct(children)
        constructor = getattr(constr, type)
        return constructor(**kwargs)
    else:
        TypeError(f'{type=} is of unknown type only dataclass, str, list, and dict is allowed!')

# def dump(obj):
#     if isinstance(obj, list):
#         return [dump(o) for o in obj]
    
#     assert isinstance(obj, dict)
#     return {k:_serialize(v) for k, v in obj.items()}




if __name__ == "__main__":
    mysection = constr.markdown('# Introduction')

