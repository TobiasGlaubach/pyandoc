import copy
import os
import re
import tempfile
from typing import Dict
import io, datetime


def get_page_title(docname):
    docname = next(iter(docname)) if not isinstance(docname, str) else docname
    return 'wikidoc_' + docname.replace('.', '_').replace(' ', '_').replace('|', '_')
    


def path2attachment(path, filename):
    return {"path" : path, "filename" : filename, "content_type" : "application/octet-stream"}




def upload_report_to_redmine(doc, redmine, project_id, report_name=None, page_title=None, force_overwrite=False, verb=True):
    """Uploads a report generated from a DocBuilder object to a Redmine wiki page.

    Args:
        doc (DocBuilder): The DocBuilder object containing the report data.
        redmine (redminelib.Redmine): A Redmine connection object.
        project_id (str): The ID of the Redmine project where the report should be uploaded.
        report_name (str, optional): The name of the report. If not provided, the follwoing schema `%Y%m%d_%H%M_exported_report` will be used.
        page_title (str, optional): The title of the Redmine wiki page. If not provided, it will be derived from the report name.
        force_overwrite (bool, optional): Whether to overwrite an existing page with the same title. Defaults to False.
        verb (bool, optional): Whether to print verbose output during upload. Defaults to True.

    Returns:
        redminelib.WikiPage: The uploaded Redmine wiki page object.

    Raises:
        AssertionError: If any of the `doc`, `project_id` or `redmine` arguments is None or empty.
    """

    if not report_name:
        report_name = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M') + '_exported_report'
    
    assert doc, 'input can not be none or empty'
    assert project_id, 'project_id can not be none or empty'
    assert isinstance(project_id, (str, int)), f'project_id must be str or int but was {type(project_id)=} {project_id=}'
    assert redmine, 'redmine can not be none or empty'
    assert report_name, 'report_name can not be none or empty'


    with tempfile.TemporaryDirectory() as tmpdir:
        dc_written = doc.export_all(report_name=report_name, dir_path=tmpdir)
        s, attachments_lst = doc.to_redmine()

        attachments = []

        # write all the io.BytesIO to the temp folder and add to be uploaded
        for dc_attachment in attachments_lst:
            abspath = os.path.join(tmpdir, re.sub(r"[^a-zA-Z0-9_.-]", "", dc_attachment['filename']))
            with open(abspath, 'wb') as fp:
                fp.write(dc_attachment['path'].read())
            attachments.append(path2attachment(abspath, os.path.basename(abspath)))
        
    
        for abspath, success in dc_written.items():
            if 'redmine' in abspath:
                continue
            assert success, f'{abspath=} failed to be saved!' 
            attachments.append(path2attachment(abspath, os.path.basename(abspath)))
        
        if not page_title:
            page_title = get_page_title(report_name)

        text = f'h1. {report_name}\n\n' + s

        try:
            page = redmine.wiki_page.get(page_title, project_id=project_id, include=['attachments'])
        except Exception as err:
            if 'ResourceNotFoundError' in str(type(err)):
                page = None
            else:
                raise

        
        #print(page)
        if not page:
            is_new = True
            page = redmine.wiki_page.new()
        else:
            is_new = False
        
        if not is_new and not force_overwrite:
            a_dc = {a.filename:a for a in page.attachments}
            to_upload = []
            for at in attachments:
                # check file sizes for images... if same name and same size... skip the upload
                n_new = os.path.getsize(at.get('path'))
                filename = at.get('filename')
                ext = filename.split('.')[-1]
                if ext in 'jpg png gif webp json'.split():
                    a_old = a_dc.get(filename, None)
                    if not a_old is None:
                        
                        if n_new == a_old.filesize:
                            if verb:
                                print(f'{filename} | {n_new=} | {a_old.filesize=} | Equal?: {n_new == a_old.filesize} --> SKIPPING!')
                            continue
                        else:
                            if verb:
                                print(f'{filename} | {n_new=} | {a_old.filesize=} | Equal?: {n_new == a_old.filesize} --> UPLOADING!')
                if verb:
                    print(filename, ' --> UPLOADING!')
                to_upload.append(at)

            filenames = [f.get('filename') for f in to_upload]
            to_delete = [a for a in page.attachments if a.filename in filenames]

            
        else:
            to_upload = attachments
            to_delete = []

        if force_overwrite:
            to_delete = [a for a in page.attachments]

        for a in to_delete:
            a.delete()
            
        if is_new:
            page.project_id = project_id
            page.title = page_title
        

        page.text = text
        page.uploads = to_upload
        page.comments = f'updated at {datetime.datetime.utcnow().isoformat()}'
        page.save()

        return page.url if page else ''
