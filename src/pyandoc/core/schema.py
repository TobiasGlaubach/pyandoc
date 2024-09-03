from dataclasses import dataclass, field, is_dataclass
from collections import UserDict, UserList





class c():
    """This is the basic schema for the main building blocks for a document"""

    # @staticmethod
    # def section(children:list=None, caption=''):
    #     return {
    #         'typ': 'section',
    #         'children': [] if children is None else children,
    #         'caption': caption
    #     }
    
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
    


class FlowDoc(UserList):
    """an unordered collection of document parts to make a document (can be used like a list)"""

    def add(self, element_to_add:dict={}):
        
        assert element_to_add, 'need to give an element_to_add!'
        assert hasattr(c, element_to_add.get('typ', None)), 'the element to add is of unknown type!'
        self.append(element_to_add)
            
    def add_kw(self, typ='', **kwargs):
        assert typ, 'need to give a content type!'
        self.add(construct(typ, **kwargs))

    def dump(self):
        return [dump(v) for v in self]

class SectionedDoc(UserDict):
    """a sectioned collection of document parts to make a document (can be used like a dict)"""

    def add_section(self, caption, children:list=None):
        assert caption, 'need to give a caption!'
        assert not caption in self, f'section with {caption=} already exists in Document'
        assert children is None or isinstance(children, list), 'children must be of type list or None'
        self[caption] = [] if children is None else children

    def add(self, section_caption:str=None, element_to_add:dict={}):
        if not section_caption and len(self):
            section_caption = list(self.keys())[-1]
        
        assert section_caption, 'need to give a section_caption to add to!'
        assert element_to_add, 'need to give an element_to_add!'
        assert hasattr(c, element_to_add.get('typ', None)), 'the element to add is of unknown type!'
        if not section_caption in self:
            self.add_section(caption=section_caption)
        self[section_caption].append(element_to_add)
            
    def add_kw(self, section_caption=None, typ='', **kwargs):
        assert typ, 'need to give a content type!'
        self.add(section_caption, construct(typ, **kwargs))

    def to_flow_doc(self):
        doc = FlowDoc()
        for section_caption, section_parts in self.items():
            doc.add(c.markdown(f'# {section_caption}'))
            for part in section_parts:
                doc.add(part)
        return doc
    
    def dump(self):
        return self.to_flow_doc().dump()


    
# @dataclass()
# class csection:
#     caption:str=field(default='')
#     children: list=field(default_factory=list)
#     type: str=field(default='section')

# @dataclass()
# class cMarkdown:
#     children:str=field(default='')
#     type: str=field(default='markdown')

# @dataclass()
# class cLargeText:
#     children:str=field(default='')
#     type: str=field(default='text')

# @dataclass()
# class cIterator:
    
#     children: list=field(default_factory=list)
#     type: str=field(default='iter')

# @dataclass()
# class cTable:
    
#     layout:str=field(default='')
#     hlines:int=field(default=1)
#     caption:str=field(default='')
    
#     type:str=field(default='table')
#     children:list=field(default_factory=lambda : [[]])

# @dataclass()
# class cImage:
    
#     children:str=field(default='')
#     imageblob:str=field(default='')
#     width:float=field(default=0.8)
#     caption:str=field(default='')
    
#     type:str=field(default='image')


#     def get_src(self):
#         s = self.imageblob.decode("utf-8") if isinstance(self.imageblob, bytes) else self.imageblob
#         if not s.startswith('data:image'):
#             s = 'data:image/png;base64, ' + s
#         return s
        

# @dataclass()
# class cVerbatim:
#     children:str=field(default='')
#     type:str=field(default='verbatim')




def _construct(v):
    if test_is_dataclass(v):
        return v
    elif isinstance(v, str):
        return v
    elif isinstance(v, list):
        return [_construct(vv) for vv in v]
    elif isinstance(v, dict):
        return construct(**v)
    else:
        TypeError(f'{type(v)=} is of unknown type only dataclass, str, list, and dict is allowed!')

def construct(type:str, **kwargs):
    assert isinstance(type, str)
    if not kwargs and not hasattr(c, type):
        return type
    elif hasattr(c, type):
        children = kwargs.get('children')
        if children:
            kwargs['children'] = _construct(children)
        constructor = getattr(c, type)
        return constructor(**kwargs)
    else:
        TypeError(f'{type=} is of unknown type only dataclass, str, list, and dict is allowed!')

def _serialize(v):
    if test_is_dataclass(v):
        return dump(v)
    elif isinstance(v, str):
        return v
    elif isinstance(v, list):
        return [dump(vv) for vv in v]
    elif isinstance(v, dict):
        return v
    else:
        TypeError(f'{type(v)=} is of unknown type only dataclass, str, list, and dict is allowed!')


def dump(obj):
    if isinstance(obj, list):
        return [dump(o) for o in obj]
    
    if not isinstance(obj, dict) and test_is_dataclass(obj):
        obj = obj.asdict()

    assert isinstance(obj, dict)
    return {k:_serialize(v) for k, v in obj.items()}

def test_is_dataclass(obj):
    return isinstance(obj, object) and is_dataclass(obj)


if __name__ == "__main__":

    mysection = c.markdown('# Introduction')
    assert test_is_dataclass(mysection)
