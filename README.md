# pydocmaker

a minimal python document maker to create reports in the following formats:

- `md`: Markdown
- `html`: HTML
- `json`: JSON
- `docx`:: Word docx
- `textile`: Textile Markup language (with images as attachments)
- `ipynb`: Jupyter/ IPython Notebooks
- `tex`: Latex Documents (with external images)
- `redmine`: Textile Markup language ready for uplaod to Redmine 


Written in pure python. 


## Installation

Install via:

```bash
pip install pydocmaker
```


## TL;DR;


Minimal Usage Example:

```python

import pydocmaker as pyd

doc = pyd.DocBuilder() # basic doc where we always append to the end
doc.add('dummy text') # adds raw text

# this is how to add preformatted
doc.add_kw('verbatim', """def hello_world():
    print('hello world!')
""")

# this is how to add markdown
doc.add_kw('markdown', """This is some *fancy* `markdown` **text**: 
- first
    - some [link]("https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png")
- second
""")

# this is how to add an image from link
doc.add_image("https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png", caption='', children='', width=0.8)

doc.show()
```

export via:

```python
    # returns string
    text_html = doc.export('html')
    # or write a file
    doc.export('path/to/my_file.html')
```

```python
doc.to_html('path/to/my_file.html') # will write a HTML file
doc.to_markdown('path/to/my_file.md') # will write a Markdown file
doc.to_docx('path/to/my_file.docx') # will write a docx file
doc.to_textile('path/to/my_file.textile.zip') # will pack all textile files and write them to a zip archive
doc.to_tex('path/to/my_file.tex.zip') # will pack all tex files and write them to a zip archive
doc.to_ipynb('path/to/my_file.ipynb') # will write a ipynb file

doc.to_json('path/to/doc.json') # saves the document
```

upload to redmine via:

```python
import redminelib
redmine = redminelib.Redmine('https://your-redmine-instance.com', key='your_redmine_token')
page = doc.to_redmine_upload(redmine, 'your-test-project')
```



## Detailed Usage Instructions

Given here is a brief overview. See the ipython notebooks within the `examples` folder within this repository for more detailed usage examples. 

## Document Builder

The `DocBuilder` class from `pydocmaker` is the basic building element for making a report. Here each element will be appended to the end of the document if no `index` or `chapter` is given. Alternatively the chapter to which to append a document part can be specified by `chapter='xxx'`. Furthermore you can also specify the index position (after which part of the document to insert) by adding `index=i` where `i` is `int`. You can use the object like a list.



## Document Parts and Schema for them

The basic building blocks for a document are called `document parts` and are always either of type `dict` or type `str` (A string will automatically parsed as a text dict element). 

Each document part has a `typ` field which states the type of document part and a `children` field, which can be either `string` or `list`. This way hirachical documents can be build if needed. 

The `document-parts` are:
- `text`: holds text as string (`children`) which will inserted directly as raw text
- `markdown`: holds text as string (`children`) which will be rendered by markdown markup language before parsing into the documents
- `image`: holds all needed information to render an image in a report. The image data is saved as a string in base64 encoded format in the `imageblob` field. A `caption` (str) can be given which will be inserted below the image. The filename is given by the `children` field. The relative width can be given by the `width` field (float). 
- `verbatim`: holds text as string (`children`) which will be inserted as preformatted text into the documents
- `iter`: a meta `document-part` which holds n sub `document-parts` in the `children` field which will be rendered and inserted into the documents in given order. 

An example of the whole schema is given below.

```json
{
  "text":     {"typ": "text", "children": ""},
  "markdown": {"typ": "markdown", "children": ""},
  "image":    {"typ": "image", "children": "", "imageblob": "", "caption": "", "width": 0.8},
  "verbatim": {"typ": "verbatim", "children": ""},
  "iter":     {"typ": "iter", "children": [] }
}
```


### Adding document parts to a doc

Alternatively you can add elements to a document directly using the add and add_kw methods of the document builders:


```python

import pydocmaker as pyd

doc = pyd.DocBuilder()
doc.add('dummy text')
doc.add({"typ": "markdown","children": "some dummy markdown text!"})
doc.add_kw('verbatim', 'this text will be shown preformatted!')
doc.add_image('https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png', caption='', children='', width=0.8)

doc.show()

```

You can also combine the two:

```python

import pydocmaker as pyd
import numpy as np

doc = pyd.DocBuilder()
doc.add('dummy text')

doc.show()

```

### Working with Images

Image from pyplot figure

```python
doc = pyd.DocBuilder()
doc.add(pyd.constr.image_from_fig(caption='test figure', fig=fig))
```

Image from link

```python
doc = pyd.DocBuilder()
doc.add(pyd.constr.image_from_link("https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png"))
```

Image from numpy array 

```python
import numpy as np
m = np.array([np.arange(255).astype(np.uint8).tolist() for i in range(255)], dtype=np.uint8)
doc = pyd.DocBuilder()
doc.add(pyd.constr.image_from_obj(m, caption = 'numpy generated image', width=0.8, name=None))
```

### Adding to Specific Chapters

as said above you can also add elements to specific chapters or locations. below is an example


```python
doc = pyd.DocBuilder()

# this will add a section 'Introduction'
doc.add_section('Introduction')

# now I can add / access the section (which is a DocBuilder) direcly like a dict
doc.add('dummy text which will be added to the introduction', chapter='Introduction')

# this will add the section 'Weather Info' and add a markdown element to it
doc.add_kw('markdown', 'This is my fancy `markdown` text for the Second Chapter', 
           chapter='Second Chapter')

# I can also add parts to the Introduction like this
doc.add_kw('markdown', 'This text will be appended to the first section after the 2nd element (`index` is zero based!), which is the text (the chapter definition itself is the 1st element!)', 
           index=1)

# and like this
doc.add_image("https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png", 
              chapter='Second Chapter')


doc.show()
```


### Upload to Redmine as Wiki Page

This is how to upload to redmine (assumes `doc` exists as generated in any of the above examples)

```python
import redminelib, datetime

text_textile, attachments_lst = doc.to_redmine()
redmine = redminelib.Redmine('https://your-redmine-instance.com', key='my_token')

page = redmine.wiki_page.new()
page.project_id = 'myproject'
page.title = 'My Wiki Page'

page.text = text_textile
page.uploads = attachments_lst
page.comments = f'updated at {datetime.datetime.utcnow().isoformat()}'
page.save()
```

### Constructing Document Parts using the `constr` factory

`document-parts` are under the hood constructed using the `constr` class which is basically a factory for `document-parts`

```python

import pydocmaker as pyd
import numpy as np

docpart = pyd.constr.markdown(children='')
docpart = pyd.constr.text(children='')
docpart = pyd.constr.verbatim(children='')
docpart = pyd.constr.iter(children=[])
docpart = pyd.constr.image(imageblob='', caption='', children='', width=0.8)
docpart = pyd.constr.image_from_link(url='https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png', caption='', children='', width=0.8)
docpart = pyd.constr.image_from_file(path='path/to/your_file.png', children='', caption='', width=0.8)
docpart = pyd.constr.image_from_fig(caption='', width=0.8, name=None, fig=None)
docpart = pyd.constr.image_from_obj(np.array(np.arange(255).tolist() * 255, dtype="uint8"), caption = '', width=0.8, name=None)
```

you can also combine adding and the functionality from above:

```python 
import pydocmaker as pyd
doc.add(pyd.constr.image_from_link("https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png"))
```
