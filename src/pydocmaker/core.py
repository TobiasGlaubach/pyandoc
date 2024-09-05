from dataclasses import dataclass, field, is_dataclass
from collections import UserDict, UserList
import json, io
import os
import re
import time
from typing import List, SupportsRead
import zipfile
import requests
import base64
import copy

import subprocess
import os




from .exporters.ex_docx import convert as _to_docx
from .exporters.ex_html import convert as _to_html
from .exporters.ex_ipynb import convert as _to_ipynb
from .exporters.ex_redmine import convert as _to_textile
from .exporters.ex_tex import convert as _to_tex


def is_notebook() -> bool:
    try:
        import __main__ as main
        return not hasattr(main, '__file__')
    except Exception as err:
        pass

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



def _is_chapter(dc):
    if not isinstance(dc, dict):
        return ''
    if not dc.get('typ') == 'markdown':
        return ''
    lines = dc.get('children', '').split('\n')
    if not len(lines) == 1:
        return ''
    if not lines[0].startswith('# '):
        return ''
    return lines[0].lstrip('#').strip()

    

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

class DocBuilder(UserList):
    """a collection of document parts to make a document (can be used like a list)"""

    def add_chapter(self, chapter_name:str, chapter_index=None):
        """Adds a new chapter to the document.

        Args:
            chapter_name (str): The name of the new chapter.
            chapter_index (int, optional): The index after which chapter to insert the new chapter. If None, appends to the end.

        Raises:
            AssertionError: If `chapter_name` is not a string or is empty.
        """
        assert isinstance(chapter_name, str), f'chapter name must be type string but was {type(chapter_name)=} {chapter_name=}'
        assert chapter_name, 'chapter_name can not be empty'
        self.add_kw('markdown', '# ' + chapter_name, chapter=chapter_index)


    def get_chapters(self, as_ranges=False):
        """Extracts chapters from the internal data and returns them as either dictionaries or ranges.

        This method iterates through the internal data structure (represented by `self.data`) and identifies chapters based on a custom logic implemented in the `_is_chapter` function.

        Args:
            as_ranges (bool, optional): If True, returns chapters as dictionaries with keys as chapter names (obtained from the previous chapter) and values as ranges of indices (inclusive-exclusive) within the data list representing the chapter content. Defaults to False.

        Returns:
            dict or dict[str, range]:
                If `as_ranges` is False, returns a dictionary where keys are chapter names and values are corresponding chapter content extracted from the data list using the identified ranges.
                If `as_ranges` is True, returns a dictionary where keys are chapter names and values are ranges of indices (inclusive-exclusive) within the data list representing the chapter content.
        """

        chapters = {}
        sec_name = ''
        i_low = 0
        for i, part in enumerate(self.data):
            if _is_chapter(part) and i > i_low:
                chapters[sec_name] = range(i_low, i)
                i_low = i

        if not as_ranges:
            return {k:self.data[rng] for k, rng in chapters.items()}
        else:
            return chapters
        

    def add(self, part:dict=None, index=None, chapter=None):
        """Appends a new document part to the given location or end of this document.

        Args:
            part (dict): The part to add. See the `constr` class for all possible parts.
            index (int, optional): The index where to insert the part. If None, appends to the end.
            chapter (str | int, optional): The chapter name or index where to insert the part. If None, appends to the end.

        Raises:
            ValueError: If the `part` is invalid, or if both `index` and `chapter` are specified.
            AssertionError: If `index` is not an integer or is out of bounds.
        """
        assert part, 'need to give an element_to_add!'
        
        if isinstance(part, str):
            part = constr.text(part)

        assert hasattr(constr, part.get('typ', None)), 'the part to add is of unknown type!'
        assert index is None or chapter is None, f'can either give index OR chapter!'

        if not chapter is None:
            sections = self.get_chapters(as_ranges=True)
            if isinstance(chapter, int):
                chapter = list(sections.keys())[chapter] 

            if not chapter in sections:
                self.add_section(chapter)
                chapter = None # just append to end!
            else:
                index = sections[chapter].stop # set to after the last element of this chapter

        if index is None:
            index = len(self) # append to end
        
        assert isinstance(index, int), f'index must be None or int but was {type(index)=} {index=}'
        assert 0 <= index <= len(self), f'index must be 0 <= index <= len(self) but was {index=}, {len(self)=}'    
        self.insert(index, part)



    def add_kw(self, typ, children=None, index=None, chapter=None, **kwargs):
        """add a document part to this document with a given typ

        Args:
            typ (str, optional): one of the allowed document part types. Either 'markdown', 'verbatim', 'text', 'iter' or 'image'.
            children (str or list): the "children" for this element. Either text directly (as string) or a list of other parts
            index (int, optional): The index where to insert the part. If None, appends to the end.
            chapter (str | int, optional): The chapter name or index where to insert the part. If None, appends to the end.

            kwargs: the kwargs for such a document part
        """
        assert typ, 'need to give a content type!'
        self.add(construct(typ, children=children, index=index, chapter=chapter, **kwargs))
    
    def add_image(self, image, caption = '', width=0.8, children=None, index=None, chapter=None, **kwargs):
        """add a image type dict from given image input.
        image can be of type:
            - pyplot figure
            - link to download an image from
            - filelike
            - numpy NxMx1 or NxMx3 matrix
            - PIL image

        Args:
            im (np.array): the image as NxMx
            caption (str, optional): the caption to give to the image. Defaults to ''.
            width (float, optional): The width for the image to have in the document. Defaults to 0.8.
            name (str, optional): A specific name/id to give to the image (will be auto generated if None). Defaults to None.
            index (int, optional): The index where to insert the part. If None, appends to the end.
            chapter (str | int, optional): The chapter name or index where to insert the part. If None, appends to the end.

        """

        if isinstance(image, str) and image.startswith('http'):
            docpart = constr.image_from_link(url=image, caption=caption, children=children, width=width)
        elif isinstance(image, str) and len(image) < 5_000 and os.path.exists(image):
            docpart = constr.image_from_file(path=image, caption=caption, children=children, width=width)
        elif isinstance(image, str):
            docpart = constr.image(imageblob=image, caption=caption, children=children, width=width)
        elif 'Figure' in str(type(image)):
            docpart = constr.image_from_fig(fig=image, caption=caption, children=children, width=width)
        else:
            docpart = constr.image_from_obj(image, caption=caption, children=children, width=width)

        self.add(docpart, index=index, chapter=chapter)

    def dump(self):
        """dump this document to a basic list of dicts for document parts

        Returns:
            list: the individual parts of the document
        """
        return [copy.deepcopy(v) for v in self]
    
    def _ret(self, m, path_or_stream):
        if isinstance(path_or_stream, str):
            with open(path_or_stream, "w") as f:
                f.write(m)
            return True
        
        if isinstance(path_or_stream, bytes):
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
    
    def to_ipynb(self, path_or_stream=None) -> str:
        """
        Converts the current object to an ipynb (iPython notebook) file.

        Args:
            path_or_stream (str or io.IOBase, optional): The path to save the file to, or a file-like object to write the data to. If not provided, the data will be returned as string.

        Returns:
            str: The data as string, or True if the data was saved successfully to a file or stream.
        """
        return self._ret(_to_ipynb(self.dump()), path_or_stream)
    
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

def _construct(v):

    if isinstance(v, str):
        return v
    elif isinstance(v, list):
        return [_construct(vv) for vv in v]
    elif isinstance(v, dict):
        return construct(**v)
    else:
        TypeError(f'{type(v)=} is of unknown type only dataclass, str, list, and dict is allowed!')

def construct(typ:str, **kwargs):
    assert isinstance(typ, str)
    if not kwargs and not hasattr(constr, typ):
        return typ
    elif hasattr(constr, typ):
        children = kwargs.get('children')
        if children:
            kwargs['children'] = _construct(children)
        constructor = getattr(constr, typ)
        return constructor(**kwargs)
    else:
        TypeError(f'{typ=} is of unknown type only dataclass, str, list, and dict is allowed!')


def load(doc:List[dict]|str|SupportsRead[str|bytes]):
    """Loads a document from a list of dictionaries, a file path, or a stream-like object.

    Args:
        doc (List[dict] | str | SupportsRead[str | bytes]): The document data, file path, or stream-like object.

    Returns:
        DocBuilder: A DocBuilder object representing the loaded document.

    Raises:
        ValueError: If the document is not a list, file path, or stream-like object, or if the file or stream cannot be loaded.
    """

    if isinstance(doc, str) and doc.strip().startswith('['):
        doc = json.loads(doc)

    if isinstance(doc, str):
        with open(doc, 'r') as fp:
            doc = json.load(fp)
    
    if hasattr(doc, 'read') and hasattr(doc, 'seek'):
        doc = json.load(fp)

    assert isinstance(doc, list), f'doc must be list but was {type(doc)=} {doc=}'
    return DocBuilder(doc)

    

    
    


def print_to_pdf(file_path, output_pdf_path):
    """Prints a file to a PDF file using the appropriate platform-specific command.

    Args:
        file_path (str): The path to the file to print.
        output_pdf_path (str): The path to the output PDF file.

    Raises:
        ValueError: If the platform is not supported.
    """

    os_name = os.name
    if os_name == "nt":
        command = ["print", "/D", "file:///dev/stdout", "/o", f"output-file={output_pdf_path}", file_path]
    elif os_name == "posix":
        command = ["lp", "-d", "file:///dev/stdout", "-o", f"output-file={output_pdf_path}", file_path]
    else:
        raise ValueError(f"Unsupported platform: {os_name}")

    subprocess.run(command, check=True)
    
# def dump(obj):
#     if isinstance(obj, list):
#         return [dump(o) for o in obj]
    
#     assert isinstance(obj, dict)
#     return {k:_serialize(v) for k, v in obj.items()}



