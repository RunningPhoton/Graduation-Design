import re

all_in = []
all_out = []
with open('anna.txt') as f:
    for line in f:
        in_txt = re.findall('[a-zA-Z]+', line)
        for word in in_txt:
            all_in.append(word.lower()+word.lower()+word.lower())
    all_in = list(set(all_in))
    for word in all_in:
        all_out.append(''.join(sorted(word)))

with open('in2.txt', 'w') as f:
    for word in all_in:
        f.write(word + '\n')
with open('out2.txt', 'w') as f:
    for word in all_out:
        f.write(word + '\n')