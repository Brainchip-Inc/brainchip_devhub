"""Unit tests for brainchip_utils.notebooks output tidying (no kernel or extras required)."""
import copy

from brainchip_utils.notebooks import noise_findings, tidy_outputs


def stream(name, text):
    return {'output_type': 'stream', 'name': name, 'text': text}


def notebook(*outputs, metadata=None):
    return {'cells': [
        {'cell_type': 'markdown', 'metadata': {}, 'source': 'Progress: 50%\r'},
        {'cell_type': 'code', 'metadata': metadata or {}, 'source': 'run()', 'outputs': list(outputs)},
    ]}


def outputs_of(nb):
    return nb['cells'][1]['outputs']


def test_progress_bar_chunks_collapse_to_the_final_state():
    nb = notebook(stream('stderr', '\rEvaluating:   0%|   | 0/3'),
                  stream('stderr', '\rEvaluating:  33%|█  | 1/3'),
                  stream('stderr', '\rEvaluating: 100%|███| 3/3'),
                  stream('stdout', 'accuracy: 0.88\n'),
                  stream('stderr', '\n'))
    assert outputs_of(tidy_outputs(nb)) == [stream('stderr', 'Evaluating: 100%|███| 3/3'),
                                            stream('stdout', 'accuracy: 0.88\n'),
                                            stream('stderr', '\n')]


def test_keras_epoch_lines_keep_their_final_state_and_crlf_keeps_text():
    nb = notebook(stream('stdout', ['Epoch 1/2\n', '1/6 [===>....]', '\r6/6 [==========] - loss: 0.6\n']),
                  stream('stdout', 'done\r\n'))
    assert outputs_of(tidy_outputs(nb)) == [
        stream('stdout', 'Epoch 1/2\n6/6 [==========] - loss: 0.6\ndone\n')]


def test_tf_log_lines_are_dropped_and_other_stderr_kept():
    nb = notebook(stream('stderr', [
        '2026-10-09 13:25:40.937067: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] '
        'Unable to register cuFFT factory\n',
        'WARNING: All log messages before absl::InitializeLog() is called are written to STDERR\n',
        'E0000 00:00:1791545140.950426  577419 cuda_dnn.cc:8579] Unable to register cuDNN factory\n',
        'W0000 00:00:1791545140.964619  577419 computation_placer.cc:177] computation placer already registered\n',
        'UserWarning: a real warning the reader should see\n']))
    assert outputs_of(tidy_outputs(nb)) == [stream('stderr', 'UserWarning: a real warning the reader should see\n')]


def test_stream_left_empty_is_removed():
    nb = notebook(stream('stderr', 'I0000 00:00:1791545140.950426  577419 x.cc:1] info\n'))
    assert outputs_of(tidy_outputs(nb)) == []


def test_streams_brought_together_by_a_dropped_chunk_are_merged():
    nb = notebook(stream('stdout', 'Epoch 1\n'),
                  stream('stderr', 'W0000 00:00:1791545140.964619  577419 x.cc:1] noise\n'),
                  stream('stdout', 'Epoch 2\n'))
    assert outputs_of(tidy_outputs(nb)) == [stream('stdout', 'Epoch 1\nEpoch 2\n')]


def test_rich_outputs_markdown_and_order_are_untouched():
    image = {'output_type': 'display_data', 'data': {'image/png': 'iVBORw0KGgo=\r'}, 'metadata': {}}
    nb = notebook(stream('stdout', 'a\n'), image, stream('stdout', 'b\n'))
    before = copy.deepcopy(nb)
    tidy_outputs(nb)
    assert outputs_of(nb) == outputs_of(before)
    assert nb['cells'][0] == before['cells'][0]


def test_timing_metadata_is_removed():
    nb = notebook(metadata={'execution': {'iopub.status.busy': '2026-10-09T11:25:40Z'}, 'tags': ['x']})
    assert tidy_outputs(nb)['cells'][1]['metadata'] == {'tags': ['x']}


def test_noise_findings_flags_headless_output_and_nothing_once_tidied():
    nb = notebook(stream('stderr', '\rstep 1'), stream('stderr', '\rstep 2'),
                  metadata={'execution': {'iopub.status.busy': '2026-10-09T11:25:40Z'}})
    problems = [problem for _, problem in noise_findings(nb)]
    assert problems == ['nbclient timing metadata', 'stream output split into 2 chunks',
                        'progress-bar residue (\\r overwrites not applied)']
    assert list(noise_findings(tidy_outputs(nb))) == []
