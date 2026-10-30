import os

# Generate a sequential filename
def mk_fname(header='test', extension='csv', directory='.'):
    ext = str(extension).lstrip('.')
    n = 1
    while True:
        fname = f"{header}_{n:02d}.{ext}"
        if not os.path.exists(os.path.join(directory, fname)):
            return fname
        n += 1

# example
# print(mk_fname('head'))
# print(mk_fname(extension='txt'))
# print(mk_fname('head', 'ext'))

# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
def sep(char='_/', n=75):
    print(char * n)


# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
def style(text, bold=False, underline=False, color=None):
    codes = []
    if bold:      codes.append('1')
    if underline: codes.append('4')
    if color:     codes.append({'red': '31', 'green': '32', 'blue': '34'}[color])
    return f"\033[{';'.join(codes)}m{text}\033[0m" if codes else str(text)

#print(style('Missing table', bold=True, underline=True))
#print(style('Warning', bold=True, color='red'))

# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/

if __name__ == '__main__':
    # python kutils.py で直接実行したときだけ動く動作確認
    print('Generated filename: ', mk_fname('header', 'ext'))