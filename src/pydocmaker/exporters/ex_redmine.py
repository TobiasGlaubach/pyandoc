import yaml, base64, time, io, copy, json, traceback, hashlib, markdown, re
from typing import List

def doc2attachment(doc_dc):
    content = io.BytesIO(bytes(json.dumps(doc_dc, indent=2), 'ascii'))
    return {"path" : content, "filename" : 'doc_dict.json', "content_type" : "application/octet-stream"}

def bindoc2attachment(mybytes, filename):
    content = io.BytesIO(mybytes)
    return {"path" : content, "filename" : filename, "content_type" : "application/octet-stream"}


def convert(doc:List[dict], with_attachments=True, files_to_upload=None):

    files_to_upload = {} if not files_to_upload else files_to_upload

    formatter = DocumentRedmineFormatter()
    s = formatter.digest(doc)
    text = '\n'.join(s)
    if with_attachments:
        attachments = [doc2attachment(doc)]
        attachments += formatter.attachments 
        attachments += [bindoc2attachment(bts, k) for k, bts in files_to_upload.items()]
        return text, attachments
    else:
        return text
    

def im2attachment(dc_img):
    mapper = {
            '/' : 'jpg',
            'i' : 'png',
            'R' : 'gif',
            'U' : 'webp'
        }
    filename = dc_img.get('children', None)
    imageblob = dc_img.get('imageblob')

    if not filename: 
        if ';base64' in imageblob:
            ext = imageblob.split(';base64')[0].split('/')[-1]
        elif mapper.get(imageblob[0], None):
            ext = mapper.get(imageblob[0], None)
        else:
            raise KeyError('extension could not be determined from imageblob')

        #filename = f'img_{time.time_ns()}_{str(id(dc))[-4:]}.{ext}'
        filename = f"img_{hashlib.md5(imageblob.encode('utf-8')).hexdigest()}.{ext}"
    description = dc_img.get('caption', '')
    if not description: 
        description = filename
    
    content = io.BytesIO(base64.b64decode(imageblob))

    return {"path" : content, "filename" : filename, "content_type" : "application/octet-stream", "description": description}


class DocumentRedmineFormatter:

    def __init__(self, out_format='textile') -> None:
        self.attachments = []
        self.out_format = out_format

    def handle_error(self, err, el) -> list:
        txt = 'ERROR WHILE HANDLING ELEMENT:\n{}\n\n'.format(el)
        if not isinstance(err, str):
            txt += '\n'.join(traceback.format_exception(err, limit=5)) + '\n'
        else:
            txt += err + '\n'
        txt = f"""<pre>\n{txt}\n</pre>"""

        return [txt]

    def digest_text(self, children='', **kwargs) -> list:
        return [children]

    def digest_markdown(self, children='', **kwargs) -> list:
        return [children]
    
    def digest_image(self, **kwargs) -> list:
        attachment = im2attachment(kwargs)

        filename = attachment.get('filename')
        caption = attachment.get('description')
        self.attachments.append(attachment)
        s = f'!{filename}({caption})!\n**IMAGE:** attachment:"{filename}" {caption}\n'
        return [s]
    
    def digest_text(self, children='', **kwargs) -> list:
        return [children]


    def digest_verbatim(self, children='', **kwargs) -> list:
        txt = self.digest(children).strip('\n')
        s = f"""<pre>{txt}</pre>"""
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
    
    def parse_md2html(self, s) -> str:
        return markdown.markdown(s, extensions=['extra', 'toc'])

    def parse_md2textile_line(self, line):
        r = re.match(r'([ \t]*)#+', line)
        n = r.group().count('#') if r else None
        if n:
            line = re.sub(r'([ \t]*)#+', rf'\1h{n}. ', line)

        r = re.match(r'^([ \t]*-{1}[ ]{1})', line)
        if r:
            g = r.group()
            line = line.replace(g, ('*'*len(g)) + ' ')

        return line

    

    def parse_md2textile(self, s) -> str:
        f = self.parse_md2textile_line
        lines = [f(line) for line in s.split('\n')]
        s = '\n'.join(lines)
        # code blocks
        s = re.sub(r"(```)([\s\S]*?)(?=```)(```)", r'<pre>\2</pre>', s)

        # links
        s = re.sub(r"(\[)([\s\S]*?)(?=\])(\])(\()([\s\S]*?)(?=\))(\))", r'"\2":\5', s)
        
        return s

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
            
            if self.out_format == 'html':
                ret = [self.parse_md2html(s) for s in ret]
            elif self.out_format == 'textile':
                ret = [self.parse_md2textile(s) for s in ret]

            return ret
        
        except Exception as err:
            return self.handle_error(err, el)

