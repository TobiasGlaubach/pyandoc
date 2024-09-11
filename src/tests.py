import sys
sys.path.insert(0, r'C:\Users\tglaubach\repos\pydocmaker\src')

import pydocmaker as pyd


doc = pyd.DocBuilder()
doc.add_chapter('Introduction')
doc.add('dummy text which will be added to the introduction')
doc.add_pre('def hello_world():\n   print("hello world!")')
doc.add_chapter('Second Chapter')
doc.add_md('This is my fancy `markdown` text for the Second Chapter')
doc.add_image("https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png")


print(doc.show())
# import os
# dir_rep = r"C:/Users/tglaubach/repos/jupyter-script-runner/src/scripts/loose_docs"
# doc_name = '20240905_1617_0_no_device_d_exported_doc'
# dc_local = doc.export_all(dir_rep=dir_rep, report_name=doc_name)
# os.listdir(dir_rep)