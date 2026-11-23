"""Render the paper's benchmark tables as standalone SVGs using LaTeX and Cairo."""
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'paper/6af734d23c7e3b92fc01fc72'
OUT = Path(__file__).resolve().parent
TABLES = {
    'ik_table': ('experiments', 'tab:ik'),
    'fk_table': ('appendix', 'tab:app:fk'),
    'trajopt_table': ('experiments', 'tab:trajopt'),
    'dynamics_table': ('experiments', 'tab:dynamics'),
}


def render():
    with tempfile.TemporaryDirectory(prefix='grim-tables-') as temp:
        work = Path(temp)
        for name, (section, label) in TABLES.items():
            source = (PAPER / 'sections' / f'{section}.tex').read_text()
            blocks = re.findall(r'\\begin\{table\*?\}(?:\[[^]]*\])?(.*?)\\end\{table\*?\}', source, re.S)
            block, = [b for b in blocks if f'\\label{{{label}}}' in b]
            block = re.sub(r'\\label\{[^}]+\}', '', block)
            block = block.replace(r'\caption{', r'\caption*{')
            tex = r'''\documentclass[border=8pt]{standalone}
\usepackage{booktabs,multirow,graphicx,caption}
\renewcommand{\arraystretch}{1.3}
\captionsetup{font=small,justification=centering}
\begin{document}
\begin{minipage}{95mm}
\captionsetup{type=table}
''' + block + r'''
\end{minipage}
\end{document}
'''
            if label == 'tab:dynamics':
                tex = tex.replace('{95mm}', '{190mm}')
            (work / f'{name}.tex').write_text(tex)
            result = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', f'{name}.tex'], cwd=work, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if result.returncode or 'Overfull' in result.stdout:
                raise RuntimeError(result.stdout)
            output = OUT / f'{name}.svg'
            subprocess.run(['pdftocairo', '-svg', str(work / f'{name}.pdf'), str(output)], check=True)
            tree = ET.parse(output)
            ns = '{http://www.w3.org/2000/svg}'
            assert tree.getroot().tag == ns + 'svg'
            assert not tree.findall('.//' + ns + 'image'), 'Expected vector-only SVG'
            print(output)


if __name__ == '__main__':
    render()
