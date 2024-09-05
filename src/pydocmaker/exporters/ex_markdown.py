import yaml, base64, time, io, copy, json, traceback, hashlib, markdown, re
from typing import List


def convert(doc:List[dict], embed_images=True):

    formatter = DocumentMarkdownFormatter(embed_images=embed_images)
    s = formatter.digest(doc)
    return '\n'.join(s)

class DocumentMarkdownFormatter:

    def __init__(self, embed_images=True) -> None:
        self.embed_images = embed_images

    def handle_error(self, err, el) -> list:
        e = str(el)
        if len(e) > 300:
            e = e[:300] + f'... (n={len(e)-300} more chars hidden)'

        txt = f'ERROR WHILE HANDLING ELEMENT:\n"{e=}"\n\n'
        if not isinstance(err, str):
            pre = '\n'.join(traceback.format_exception(err, limit=5)) + '\n'
        else:
            pre = err

        return [txt] + self.digest_verbatim(pre)

    def digest_text(self, children='', **kwargs) -> list:
        return [children]

    def digest_markdown(self, children='', **kwargs) -> list:
        return [children]
    
    def digest_image(self, **kwargs) -> list:
        
        filename = kwargs.get('filename')
        caption = kwargs.get('caption')
        imageblob = kwargs.get('imageblob')
        
        description = []
        if filename:
            description.append(str(filename))
        if caption:
            description.append(str(caption))

        if not description:
            description = f'{time.time_ns()}_embedded_image'
        else:
            description = ' '.join(description)

        lines = []
        lines.append('')

        if filename:    
            lines.append(f'*filename:* {filename}')
            lines.append('')
        
        lines.append('')

        # HACK: handle non PNG type properly!
        if self.embed_images:
            lines.append(f'![{description}](data:image/png;base64,{imageblob})')
        else:
            i = imageblob[:20] + f'... (n={len(imageblob)-20} more chars hidden))' if len(imageblob) > 20 else imageblob
            lines.append(f'#[{description}](data:image/png;base64,{i}')
        lines.append('')

        if caption:
            lines.append(f'*caption:* {filename}')
            lines.append('')

        return lines
    
    def digest_text(self, children='', **kwargs) -> list:
        return [children]


    def digest_verbatim(self, children='', **kwargs) -> list:
        if isinstance(children, str):
            txt = children.strip('\n')
        else:
            txt = self.digest(children)
        s = f"""```\n{txt}\n```"""
        return [s]


    def digest_iter(self, el) -> list:
        parts = []
        if isinstance(el, dict) and el.get('typ', '') == 'iter' and isinstance(el.get('children', None), list):
            el = el['children']
        
        assert isinstance(el, list)
        for p in el:
            parts += self.digest(p)
            parts.append('\n\n')

        return parts
    

    def digest_str(self, el) -> list:
        return [el]
        
    def digest(self, el) -> list:
        try:
            
            if not el:
                return ''
            elif isinstance(el, str):
                ret = self.digest_str(el)
            elif isinstance(el, dict) and 'typ' in el and el['typ'] == 'iter':
                ret = self.digest_iter(el)
            elif isinstance(el, list) and el:
                ret = self.digest_iter(el)
            elif isinstance(el, dict) and 'typ' in el and el['typ'] == 'image':
                ret = self.digest_image(**el)
            elif isinstance(el, dict) and 'typ' in el and el['typ'] == 'text':
                ret = self.digest_text(**el)
            elif isinstance(el, dict) and 'typ' in el and el['typ'] == 'verbatim':
                ret = self.digest_verbatim(**el)
            elif isinstance(el, dict) and 'typ' in el and el['typ'] == 'markdown':
                ret = self.digest_markdown(**el)
            else:
                return self.handle_error(f'the element of type {type(el)} {el=}, could not be parsed.')
            
            return ret
        
        except Exception as err:
            return self.handle_error(err, el)

