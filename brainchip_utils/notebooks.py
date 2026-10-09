# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""Execute notebooks headlessly and save them the way Jupyter does from a browser.

Executing with a bare nbclient call (the usual way to run a notebook from a script, tmux
or CI) stores every tqdm / Keras progress-bar refresh as a separate stream output,
records per-cell timing metadata, and keeps the TensorFlow/XLA C++ log lines the host
prints at import. This module executes a notebook, then tidies its outputs:

- consecutive stdout/stderr chunks are merged and carriage-return overwrites applied,
  so only the final state of each progress bar is stored;
- TensorFlow/XLA C++ log lines (cuFFT/cuDNN registration, computation placer, absl)
  are dropped from stderr;
- per-cell timing metadata is not recorded, and removed if present.

Usage (needs the [notebooks] extra: pip install -e ".[notebooks]"):

    python -m brainchip_utils.notebooks NOTEBOOK [NOTEBOOK ...] [--timeout S]
    python -m brainchip_utils.notebooks --tidy-only NOTEBOOK [NOTEBOOK ...]

Each notebook runs from its own directory, as it would in Jupyter. tidy_outputs() and
noise_findings() work on plain notebook dicts and use only the standard library, so
check_content.py can use them without the extra.
"""
import argparse
import pathlib
import re
import time

# C++ log lines TensorFlow/XLA print at import, e.g.
#   2026-10-09 13:25:40.937067: E external/local_xla/...] Unable to register cuFFT factory...
#   E0000 00:00:1791545140.950426  577419 cuda_dnn.cc:8579] Unable to register cuDNN factory...
_TF_LOG_LINE = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+: [IWEF] '
                          r'|[IWEF]\d{4} \d\d:\d\d:\d+\.\d+ +\d+ '
                          r'|WARNING: All log messages before absl::InitializeLog\(\) is called)')


def _text(output):
    """Stream text as one string (nbformat stores it as a list of lines on disk)."""
    text = output.get('text', '')
    return ''.join(text) if isinstance(text, list) else text


def _apply_carriage_returns(text):
    """Within each line, keep only what follows the last \\r, as a terminal would show it."""
    # Trailing \r first, so a \r\n line ending keeps its text
    return '\n'.join(line.rstrip('\r').rsplit('\r', 1)[-1] for line in text.split('\n'))


def _drop_tf_log_lines(text):
    return ''.join(line for line in text.splitlines(keepends=True) if not _TF_LOG_LINE.match(line))


def _merge_streams(outputs):
    """Join consecutive stream outputs with the same name (stdout or stderr)."""
    merged = []
    for output in outputs:
        previous = merged[-1] if merged else None
        if (output.get('output_type') == 'stream' and previous is not None
                and previous.get('output_type') == 'stream' and previous.get('name') == output.get('name')):
            previous['text'] = _text(previous) + _text(output)
        else:
            merged.append(output)
    return merged


def tidy_outputs(nb):
    """Tidy a notebook's outputs in place (see the module docstring) and return it."""
    for cell in nb['cells']:
        if cell.get('cell_type') != 'code':
            continue
        cell.get('metadata', {}).pop('execution', None)
        outputs = _merge_streams(cell.get('outputs', []))
        for output in outputs:
            if output.get('output_type') == 'stream':
                output['text'] = _apply_carriage_returns(_text(output))
                if output.get('name') == 'stderr':
                    output['text'] = _drop_tf_log_lines(output['text'])
        # Dropping a stream left empty can bring two of the same name together again
        cell['outputs'] = _merge_streams(
            [o for o in outputs if o.get('output_type') != 'stream' or o['text']])
    return nb


def noise_findings(nb):
    """Yield (cell_index, problem) for output noise a headless run leaves behind:
    split stream outputs, progress-bar \\r residue and nbclient timing metadata."""
    for index, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') != 'code':
            continue
        if 'execution' in cell.get('metadata', {}):
            yield index, 'nbclient timing metadata'
        outputs = cell.get('outputs', [])
        streams = [o for o in outputs if o.get('output_type') == 'stream']
        if any(a.get('output_type') == b.get('output_type') == 'stream' and a.get('name') == b.get('name')
               for a, b in zip(outputs, outputs[1:])):
            yield index, f'stream output split into {len(streams)} chunks'
        if any('\r' in _text(o) for o in streams):
            yield index, 'progress-bar residue (\\r overwrites not applied)'


def execute(path, timeout=1800, kernel_name='python3'):
    """Execute a notebook in place, from its own directory, and save it tidied."""
    import nbformat
    from nbclient import NotebookClient

    path = pathlib.Path(path).resolve()
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=timeout, kernel_name=kernel_name, record_timing=False,
                   resources={'metadata': {'path': str(path.parent)}}).execute()
    nbformat.write(tidy_outputs(nb), path)


def tidy(path):
    """Tidy an already-executed notebook in place, without running it."""
    import nbformat

    nb = nbformat.read(path, as_version=4)
    nbformat.write(tidy_outputs(nb), path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('notebooks', nargs='+', help='notebooks to execute (in place)')
    parser.add_argument('--timeout', type=int, default=1800, help='per-cell timeout, seconds')
    parser.add_argument('--kernel', default='python3', help='Jupyter kernel name')
    parser.add_argument('--tidy-only', action='store_true',
                        help="tidy the notebooks' existing outputs without executing them")
    args = parser.parse_args()

    for notebook in args.notebooks:
        start = time.time()
        if args.tidy_only:
            tidy(notebook)
        else:
            execute(notebook, timeout=args.timeout, kernel_name=args.kernel)
        verb = 'tidied' if args.tidy_only else 'executed'
        print(f'{verb} {notebook} in {time.time() - start:.0f} s')


if __name__ == '__main__':
    main()
