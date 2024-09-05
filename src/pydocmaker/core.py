from dataclasses import dataclass, field, is_dataclass
from collections import UserDict, UserList
import json, io
import os
import re
import tempfile
import time
from typing import List, BinaryIO, TextIO
import zipfile
import requests
import base64
import copy

import subprocess
import os



from .util import upload_report_to_redmine

from .exporters.ex_docx import convert as _to_docx
from .exporters.ex_html import convert as _to_html
from .exporters.ex_ipynb import convert as _to_ipynb
from .exporters.ex_redmine import convert as _to_textile
from .exporters.ex_markdown import convert as _to_markdown
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



def _is_chapter(dc, pre='##'):
    if not isinstance(dc, dict):
        return ''
    if not dc.get('typ') == 'markdown':
        return ''
    lines = dc.get('children', '').split('\n')
    if not len(lines) == 1:
        return ''
    if not lines[0].startswith(pre + ' '):
        return ''
    return lines[0].lstrip(pre).strip()

    

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

        if not children:
            # HACK: need to get format somehow
            children = f'img_{time.time_ns()}.png'

        return {
            'typ': 'image',
            'children': re.sub(r"[^a-zA-Z0-9_.-]", '', children),
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
        

    def image_from_fig(caption='', width=0.8, children=None, fig=None, **kwargs):
        """convert a matplotlib figure (or the current figure) to a document image dict to later add to a document

        Args:
            caption (str, optional): the caption to give to the image. Defaults to ''.
            width (float, optional): The width for the image to have in the document. Defaults to 0.8.
            children (str, optional): A specific name/id to give to the image (will be auto generated if None). Defaults to None.
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
        
        if children is None:
            id_ = str(id(img))[-2:]
            children = f'figure_{int(time.time())}_{id_}.png'

        return constr.image(imageblob = 'data:image/png;base64,' + img, children=children, caption=caption, width=width)


    @staticmethod
    def image_from_obj(im, caption = '', width=0.8, children=None):
        """make a image type dict from given image of type matrix, filelike or PIL image

        Args:
            im (np.array): the image as NxMx
            caption (str, optional): the caption to give to the image. Defaults to ''.
            width (float, optional): The width for the image to have in the document. Defaults to 0.8.
            children (str, optional): A specific name/id to give to the image (will be auto generated if None). Defaults to None.

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
            if not children:
                children = os.path.basename(im)            
            im = open(im, 'rb')

        # file like --> make bytes
        if hasattr(im, 'read'):
            im.seek(0)   
            im = im.read()
        
        # bytes --> make b64 string
        if isinstance(im, bytes):
            im = base64.b64encode(im).decode('utf-8')

        if children is None:
            id_ = str(id(im))[-2:]
            children = f'image_{int(time.time())}_{id_}.png'

        return constr.image(imageblob = 'data:image/png;base64,' + im, children=children, caption=caption, width=width)

buildingblocks = 'text markdown image verbatim iter'.split()

class DocBuilder(UserList):
    """a collection of document parts to make a document (can be used like a list)"""

    export_engines = ['md', 'html', 'json', 'docx', 'textile', 'ipynb', 'tex', 'redmine']
    export_engine_extensions = {
        'md': '.md', 
        'html':'.html', 
        'json':'.json', 
        'docx': '.docx',
        'textile': '.textile.zip',
        'ipynb': '.ipynb', 
        'tex': '.tex.zip'
    }

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
        chapters = list(self.get_chapters().keys())
        assert chapter_name not in chapters, f'chapter with {chapter_name=} already exists in document {chapters=}!'
        self.add_kw('markdown', '## ' + chapter_name, chapter=chapter_index)

    def get_chapter(self, chapter) -> List[dict]:
        """Retrieves a specific chapter from the document.

        Args:
            chapter (str): The name of the chapter to retrieve.

        Returns:
            List[dict]: A list of dictionaries representing the content of the specified chapter.
        """
        return self.get_chapters(as_ranges=False)[chapter]
    
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
        last_chap_name = ''
        i_low = 0
        for i, part in enumerate(self.data):
            i_chap_name = _is_chapter(part)
            if i_chap_name and i == 0 and i_low == 0:
                last_chap_name = i_chap_name

            if i_chap_name and i >= i_low:
                chapters[last_chap_name] = slice(i_low, i)
                i_low = i
                last_chap_name = i_chap_name

        if last_chap_name and i >= i_low and not last_chap_name in chapters:
            chapters[last_chap_name] = slice(i_low, i)
            
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
            chapters = self.get_chapters(as_ranges=True)
            if isinstance(chapter, int):
                chapter = list(chapters.keys())[chapter] 

            if not chapter in chapters:
                self.add_chapter(chapter)
                chapter = None # just append to end!
            else:
                index = chapters[chapter].stop + 1 # set to after the last element of this chapter

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
        self.add(construct(typ, children=children, **kwargs), index=index, chapter=chapter)
    
    
    def add_fig(self, fig=None, caption = '', width=0.8, children=None, index=None, chapter=None, **kwargs):
        """add a pyplot figure type dict from given image input.
        
        Args:
            fig (matplotlib figure, optional): the figure which to upload (or the current figure if None). Defaults to None.
            caption (str, optional): the caption to give to the image. Defaults to ''.
            width (float, optional): The width for the image to have in the document. Defaults to 0.8.
            children (str, optional): A specific name/id to give to the image (will be auto generated if None). Defaults to None.
            index (int, optional): The index where to insert the part. If None, appends to the end.
            chapter (str | int, optional): The chapter name or index where to insert the part. If None, appends to the end.

        """
        self.add(constr.image_from_fig(caption=caption, width=width, children=children, fig=fig, **kwargs), index=index, chapter=chapter)
                 

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
            children (str, optional): A specific name/id to give to the image (will be auto generated if None). Defaults to None.
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
        

        if path_or_stream and isinstance(path_or_stream, str):
            mode = 'w' if isinstance(m, str) else 'wb'
            with open(path_or_stream, mode) as f:
                f.write(m)
            return True
        
        elif hasattr(path_or_stream, 'write'):
            try:
                path_or_stream.write(m)
            except TypeError:
                if isinstance(m, str):
                    path_or_stream.write(m.encode())
                else:
                    raise

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

    def to_markdown(self, path_or_stream=None, embed_images=True) -> str:
        """
        Converts the current object to a Markdown string or writes it to a file.

        Args:
            path_or_stream (str or io.IOBase, optional): The path to save the Markdown file to, or a file-like object to write the data to. If not provided, the Markdown string will be returned.
            embed_images (bool, optional): Whether to embed images as base64 strings within the Markdown. Defaults to True.

        Returns:
            str or bool: The Markdown string if `path_or_stream` is not provided, or True if the Markdown was successfully written to the file or stream.
        """
        return self._ret(_to_markdown(self.dump(), embed_images=embed_images), path_or_stream)

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
    
    def to_redmine_upload(self, redmine, project_id:str|int, report_name=None, page_title=None, force_overwrite=False, verb=True):
        """Converts the current object to a Redmine Textile like text (and attachments) and Uploads it to a Redmine wiki page.
        This will also export the document to all possible formats and attach them to the wiki page.

        Args:
            redmine (redminelib.Redmine): A Redmine connection object.
            project_id (str|int): The ID of the Redmine project where the report should be uploaded.
            report_name (str, optional): The name of the report. If not provided, the follwoing schema `%Y%m%d_%H%M_exported_report` will be used.
            page_title (str, optional): The title of the Redmine wiki page. If not provided, it will be derived from the report name.
            force_overwrite (bool, optional): Whether to overwrite an existing page with the same title. Defaults to False.
            verb (bool, optional): Whether to print verbose output during upload. Defaults to True.

        Returns:
            redminelib.WikiPage: The uploaded Redmine wiki page object.

        Raises:
            AssertionError: If any of the `doc`, `project_id` or `redmine` arguments is None or empty.
        """
            
        return upload_report_to_redmine(self, redmine=redmine, project_id=project_id, report_name=report_name, page_title=page_title, force_overwrite=force_overwrite, verb=verb)
    
    def to_pdf(self, path_or_stream=None):
        """Exports the document to a PDF file.

        Args:
            output_pdf_path (str, optional): The path to save the PDF file to. If not provided, a temporary file will be used.

        Returns:
            str: The path to the exported PDF file.
        """
        os_name = os.name
        assert os_name == 'posix', 'only posix like operation systems are supported for printing a pdf file!'

        with tempfile.TemporaryDirectory() as tmpdir:
            html_file_path = os.path.join(tmpdir, "temp.html")
            self.to_html(html_file_path)

            if not path_or_stream or not isinstance(path_or_stream, str):
                output_pdf_path = tempfile.NamedTemporaryFile(suffix=".pdf").name
            else:
                output_pdf_path = path_or_stream

            print_to_pdf(html_file_path, output_pdf_path)

            if not path_or_stream:
                data = open(output_pdf_path, 'rb').read()
                if not len(data):
                    raise IOError(f'failed to write {output_pdf_path=}')
                return data
            elif hasattr(path_or_stream, 'write'):
                data = open(output_pdf_path, 'rb').read()
                if not len(data):
                    raise IOError(f'failed to write {output_pdf_path=}')           
                path_or_stream.write(data)
                return True
            else:
                assert isinstance(path_or_stream, str), f'path_or_stream is not a string but {type(path_or_stream)=}'
                assert isinstance(output_pdf_path, str), f'output_pdf_path is not a string but {type(output_pdf_path)=}'
                assert output_pdf_path == path_or_stream, f'something went wrong, since the PDF file was written to {output_pdf_path=} instead of {path_or_stream=}'
                exists = os.path.exists(path_or_stream)
                if not exists:
                    raise IOError(f'failed to write {path_or_stream=}')
                return exists

    def export_all(self, dir_path=None, report_name='exported_report', **kwargs):
        """
        Exports the document to all possible formats.

        Args:
            dir_path (str, optional): The path to the directory where the exported files should be saved. If not provided, a dict with the returned data from the exporters will be returned.
            report (str, optional): The base name for the exported files. Defaults to "exported_report".
            **kwargs: Additional keyword arguments specific to the chosen export formats.

        Returns:
            dict: A dictionary containing the exported data or paths for each engine.
        """
        return self.export_many(engines=None, dir_path=dir_path, report_name=report_name, **kwargs)

    def export_many(self, engines:list[str]=None, dir_path=None, report_name='exported_report', **kwargs):
        """
        Exports the document to multiple formats.

        Args:
            engines (list[str], optional): A list of export engines to use. If not provided, all supported engines will be used.
            dir_path (str, optional): The path to the directory where the exported files should be saved. If not provided, a dict with the returned data from the exporters will be returned.
            report (str, optional): The base name for the exported files. Defaults to "exported_report".
            **kwargs: Additional keyword arguments specific to the chosen export formats.

        Returns:
            dict: A dictionary containing the exported data or paths for each engine.
        """
        if engines is None and dir_path is None:
            engines = [e for e in DocBuilder.export_engines] # all engines
        else:
            engines = list(DocBuilder.export_engine_extensions.keys())

        
        if not dir_path is None:
            assert os.path.exists(dir_path), f'given {dir_path=} does not exist!'
            assert os.path.isdir(dir_path), f'given {dir_path=} is not a directory'
        
        ret = {}
        for engine in engines:
            engine = engine.strip('').strip('.')
            if dir_path is None:
                ext = DocBuilder.export_engine_extensions.get(engine, '.' + engine)
                path = None
                key = report_name + ext
            else:
                ext = DocBuilder.export_engine_extensions.get(engine, '.' + engine)
                path = os.path.join(dir_path, report_name + ext)
                key = path
            
            ret[key] = self.export(engine, path, **kwargs.get(engine, {}))
        
        return ret
    

    def export(self, engine:str, path_or_stream=None, **kwargs):
        """Exports the document to a specified format.

        Args:
            engine (str): The format to export to. Valid options are: 'md', 'markdown', 'json', 'html', 'tex', 'latex', 'textile', 'word', 'docx', and 'redmine'.
            path_or_stream (str or io.IOBase, optional): The path to save the exported file to, or a file-like object to write the data to.
            **kwargs: Additional keyword arguments specific to the chosen export format.

        Returns:
            str or bool: The exported data or True if the data was successfully written to a file or stream.

        Raises:
            KeyError: If the specified `engine` is not supported.
        """

        if '\\' in engine or '/' in engine and not path_or_stream:
            path_or_stream = engine
            engine = os.path.basename(engine)

        
        engine = engine.split('.')[-1]
        engine = engine.lower().strip()

        if engine in ['md', 'markdown']:
            return self.to_markdown(path_or_stream=path_or_stream, **kwargs)
        elif engine in ['json']:
            return self.to_json(path_or_stream=path_or_stream, **kwargs)
        elif engine in ['html']:
            return self.to_html(path_or_stream=path_or_stream, **kwargs)
        elif engine in ['pdf']:
            return self.to_pdf(path_or_stream=path_or_stream, **kwargs)
        elif engine in ['tex', 'latex']:
            return self.to_tex(path_or_stream=path_or_stream, **kwargs)
        elif engine in ['textile']:
            return self.to_textile(path_or_stream=path_or_stream, **kwargs)
        elif engine in ['ipynb', 'jupyter', 'notebook']:
            return self.to_ipynb(path_or_stream=path_or_stream, **kwargs)
        elif engine in ['word', 'docx']:
            return self.to_docx(path_or_stream=path_or_stream, **kwargs)
        elif engine in ['redmine']:
            assert not path_or_stream, 'redmine engine can not handle writing to path_or_stream!'
            return self.to_redmine(**kwargs)
        else:
            raise KeyError(f'engine must be in: {DocBuilder.export_engines=}, but was {engine=}')
        

    def show(self, index=None, chapter=None, engine='markdown'):
        """Displays the document or a specific part of it in ipython display or via print

        Args:
            index (int, optional): The index of the part to display.
            chapter (str, optional): The name of the chapter to display.

        Raises:
            AssertionError: If both `index` and `chapter` are specified.
        """
        
        assert index is None or chapter is None, f'can either give index OR chapter!'
        
        if index:
            DocBuilder([self[index]]).show()
        elif chapter:
            DocBuilder(self.get_chapter(chapter)).show()
        
        if is_notebook():
            from IPython.display import display, HTML, Markdown
            # display(HTML(self.to_html()))

        else:
            print(self.to_markdown(embed_images=False))

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
    """construct a document-part dict from the given typ and some kwargs"""
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


def load(doc:List[dict]|str|BinaryIO|TextIO|bytes):
    """Loads a document from a list of dictionaries, a file path, or a stream-like object.

    Args:
        doc (List[dict] | str | BinaryIO | TextIO]): The document data, file path, or stream-like object.

    Returns:
        DocBuilder: A DocBuilder object representing the loaded document.

    Raises:
        ValueError: If the document is not a list, file path, or stream-like object, or if the file or stream cannot be loaded.
    """
    if isinstance(doc, bytes):
        doc = doc.decode()

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
    assert os_name == 'posix', 'only posix like operation systems are supported for printing a pdf file!'

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



