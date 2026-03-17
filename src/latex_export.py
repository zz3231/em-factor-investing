"""LaTeX export utilities for tables, figures, and paper templates."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ===================================================================
# Table export
# ===================================================================

def df_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    fmt: str = '.4f',
) -> str:
    """Convert a DataFrame to a LaTeX table string with booktabs formatting.

    Parameters
    ----------
    df : pd.DataFrame
        Data to render.
    caption : str
        Table caption.
    label : str
        LaTeX label for cross-referencing (e.g. ``tab:perf``).
    fmt : str
        Numeric format specifier applied to all float columns.

    Returns
    -------
    str
        Complete LaTeX table environment string.
    """
    n_cols = len(df.columns)
    col_fmt = 'l' + 'r' * n_cols

    header = (
        '\\begin{table}[htbp]\n'
        '\\centering\n'
        f'\\caption{{{caption}}}\n'
        f'\\label{{{label}}}\n'
        f'\\begin{{tabular}}{{{col_fmt}}}\n'
        '\\toprule\n'
    )

    col_names = ' & '.join([df.index.name or ''] + list(df.columns))
    header += col_names + ' \\\\\n\\midrule\n'

    rows: list[str] = []
    for idx, row in df.iterrows():
        vals = []
        for v in row:
            if isinstance(v, float):
                vals.append(f'{v:{fmt}}')
            else:
                vals.append(str(v))
        rows.append(f'{idx} & ' + ' & '.join(vals) + ' \\\\')

    body = '\n'.join(rows)

    footer = (
        '\n\\bottomrule\n'
        '\\end{tabular}\n'
        '\\end{table}\n'
    )

    return header + body + footer


def save_latex_table(latex_str: str, filepath: str) -> None:
    """Write a LaTeX table string to a file.

    Parameters
    ----------
    latex_str : str
        LaTeX source to save.
    filepath : str
        Destination file path.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(latex_str)


# ===================================================================
# Figure export
# ===================================================================

def save_figure_for_latex(
    fig: plt.Figure,
    filename: str,
    output_dir: str,
) -> str:
    """Save a matplotlib figure as a PDF and return the includegraphics command.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    filename : str
        File name (without directory), e.g. ``'cumulative_returns.pdf'``.
    output_dir : str
        Directory in which to save the figure.

    Returns
    -------
    str
        LaTeX ``\\includegraphics`` command referencing the saved file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, format='pdf', bbox_inches='tight')
    plt.close(fig)

    return (
        f'\\begin{{figure}}[htbp]\n'
        f'\\centering\n'
        f'\\includegraphics[width=\\textwidth]{{{filename}}}\n'
        f'\\caption{{TODO: caption}}\n'
        f'\\label{{fig:{Path(filename).stem}}}\n'
        f'\\end{{figure}}'
    )


# ===================================================================
# Paper template
# ===================================================================

def generate_paper_template(output_dir: str) -> None:
    """Create a main.tex file with standard academic paper structure.

    Parameters
    ----------
    output_dir : str
        Directory in which to write ``main.tex``.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    template = r"""\documentclass[12pt,a4paper]{article}

% ------------------------------------------------------------------
% Packages
% ------------------------------------------------------------------
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage[round]{natbib}
\usepackage[margin=1in]{geometry}
\usepackage{setspace}
\usepackage{float}

\onehalfspacing

% ------------------------------------------------------------------
% Title
% ------------------------------------------------------------------
\title{Value Factor Portfolio Construction in Emerging Markets}
\author{Author Name \\ Columbia Business School}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
This paper investigates portfolio construction methodologies applied to a
value-factor long-only strategy in emerging-market equities.  We compare
equal weighting, inverse-volatility weighting, minimum-variance,
maximum Sharpe ratio, risk-parity, and hierarchical risk parity (HRP)
allocation schemes.  We evaluate each approach across multiple
performance dimensions including risk-adjusted returns, drawdowns, and
out-of-sample stability.
\end{abstract}

% ------------------------------------------------------------------
\section{Introduction}
\label{sec:intro}
% ------------------------------------------------------------------

[Introduction text here.]

% ------------------------------------------------------------------
\section{Data and Methodology}
\label{sec:data}
% ------------------------------------------------------------------

\subsection{Data}
[Description of data sources, sample period, and universe construction.]

\subsection{Factor Construction}
[Details of value factor definition and signal construction.]

\subsection{Portfolio Construction}
[Description of the six portfolio construction methods.]

% ------------------------------------------------------------------
\section{Empirical Results}
\label{sec:results}
% ------------------------------------------------------------------

\subsection{Full-Sample Performance}
[Full-sample performance comparison.]

\subsection{In-Sample vs.\ Out-of-Sample}
[IS/OOS split analysis.]

\subsection{Weight Characteristics}
[Discussion of weight concentration and turnover.]

\subsection{Drawdown Analysis}
[Maximum drawdown and recovery analysis.]

% ------------------------------------------------------------------
\section{Robustness}
\label{sec:robustness}
% ------------------------------------------------------------------

[Alternative specifications, sensitivity to bounds, look-back windows.]

% ------------------------------------------------------------------
\section{Conclusion}
\label{sec:conclusion}
% ------------------------------------------------------------------

[Summary of findings and implications.]

% ------------------------------------------------------------------
% References
% ------------------------------------------------------------------
\bibliographystyle{plainnat}
\bibliography{references}

\end{document}
"""

    filepath = os.path.join(output_dir, 'main.tex')
    with open(filepath, 'w') as f:
        f.write(template)


def generate_references(output_dir: str) -> None:
    """Create a references.bib file with key citations.

    Parameters
    ----------
    output_dir : str
        Directory in which to write ``references.bib``.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    bib = r"""@article{lopezdeprado2016,
  author  = {Lopez de Prado, Marcos},
  title   = {Building Diversified Portfolios that Outperform Out-of-Sample},
  journal = {Journal of Portfolio Management},
  year    = {2016},
  volume  = {42},
  number  = {4},
  pages   = {59--69},
}

@article{markowitz1952,
  author  = {Markowitz, Harry},
  title   = {Portfolio Selection},
  journal = {The Journal of Finance},
  year    = {1952},
  volume  = {7},
  number  = {1},
  pages   = {77--91},
}

@article{famafrench1993,
  author  = {Fama, Eugene F. and French, Kenneth R.},
  title   = {Common Risk Factors in the Returns on Stocks and Bonds},
  journal = {Journal of Financial Economics},
  year    = {1993},
  volume  = {33},
  number  = {1},
  pages   = {3--56},
}

@article{famafrench2012,
  author  = {Fama, Eugene F. and French, Kenneth R.},
  title   = {Size, Value, and Momentum in International Stock Returns},
  journal = {Journal of Financial Economics},
  year    = {2012},
  volume  = {105},
  number  = {3},
  pages   = {457--472},
}

@article{ledoitwolf2004,
  author  = {Ledoit, Olivier and Wolf, Michael},
  title   = {A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices},
  journal = {Journal of Multivariate Analysis},
  year    = {2004},
  volume  = {88},
  number  = {2},
  pages   = {365--411},
}

@article{maillard2010,
  author  = {Maillard, S\'{e}bastien and Roncalli, Thierry and Teiletche, J\'{e}r\^{o}me},
  title   = {The Properties of Equally Weighted Risk Contribution Portfolios},
  journal = {Journal of Portfolio Management},
  year    = {2010},
  volume  = {36},
  number  = {4},
  pages   = {60--70},
}

@article{demiguel2009,
  author  = {DeMiguel, Victor and Garlappi, Lorenzo and Uppal, Raman},
  title   = {Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?},
  journal = {The Review of Financial Studies},
  year    = {2009},
  volume  = {22},
  number  = {5},
  pages   = {1915--1953},
}
"""

    filepath = os.path.join(output_dir, 'references.bib')
    with open(filepath, 'w') as f:
        f.write(bib)
