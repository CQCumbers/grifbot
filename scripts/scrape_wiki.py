import re

# first manually copy and paste
# scripts from wiki to text file
PATH = 'wiki_scripts.txt'
PATH2 = 'scripts.txt'
text = open(PATH).read()
lines = []

for line in text.split('\n'):
    if len(line) == 0:
        continue
    line = line.replace('CSI: Miami', 'CSI Miami')
    if line.startswith('Episode') or 'Teaser Trailer' == line:
        lines.append(line)
        lines.append('')
        continue
    if line.startswith('Red vs. Blue'):
        lines.append('')
        lines.append('')
        lines.append(line)
        continue
    if not ':' in line:
        lines.append(line)
        continue
    speaker, speech = line.split(':', 1)
    if 'LOPEZ' in speaker.upper() and '[' in speech:
        lines.append('LOPEZ:' + re.sub(r'\[.*\]', '', speech.strip()))
        lines.append('CAPTION:' + re.search(r'\[(.*)\]', speech.strip()).group(1))
        continue
    speech = speech.replace('[','(').replace(']',')')
    lines.append(speaker.strip().upper() + ':' + speech.strip())

text = '\n'.join(lines)
replacemap = {"\x91": '"', "\x93": '"', "\x92": "'", "\x94": "'",
    '[': '(', ']': ')', '\x85': '\n', '\xa0': ' ', '\x96': ''}
for k, v in replacemap.items():
    text = text.replace(k, v)

with open(PATH2, 'a') as f:
    f.write(text)
