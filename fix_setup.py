import re
with open(r'auto_LiRPA\setup.py', 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace(
    "long_description = (this_directory / 'README.md').read_text()",
    "long_description = (this_directory / 'README.md').read_text(encoding='utf-8')"
)
with open(r'auto_LiRPA\setup.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Fixed setup.py')
