from pypdf import PdfReader

path = r'c:/Users/adele/Documents/UoB_4thYear/TM_modelling/TMM202526Lab01.pdf'
reader = PdfReader(path)
print('pages', len(reader.pages))
for i, page in enumerate(reader.pages[:12], 1):
    text = page.extract_text() or ''
    print(f'--- PAGE {i} ---')
    print(text[:4000])
