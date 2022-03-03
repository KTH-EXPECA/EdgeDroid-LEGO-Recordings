import functools
import itertools
import subprocess
from collections import defaultdict, deque
from multiprocessing import Pool
from pathlib import Path
from typing import Deque, Dict, Tuple

import cv2
import nptyping as npt
import numpy as np
import pandas as pd
from gabriel_lego.cv import image_util as img_util
from gabriel_lego.cv.lego_cv import LEGOCVError, LowConfidenceError
from gabriel_lego.lego_engine.board import BoardState
from tqdm import tqdm
from tqdm.contrib import tenumerate

remote_src = 'blizzard@Blizzard:/home/blizzard/data/experiment_logs/'
frame_path_fmt = '{run_id}/main_task/frames/processed/frame_{frame:d}.jpeg'

out_dir = Path('./frames').resolve()


def find_frames() -> Tuple[pd.DataFrame, Deque[Dict[str, int]]]:
    sqr_mappings = pd.read_csv('latinsqr_mappings.csv',
                               index_col='run_id')
    sqr_mappings = sqr_mappings['latin_sqr_nr'].to_dict()

    # find all success frames
    success_frames: Deque = deque()
    init_frames: Deque = deque()
    for run_id in (run_pbar := tqdm(sqr_mappings.keys())):
        run_pbar.set_description(f'Finding frames for run {run_id}.')

        run_steps = pd.read_csv(f'./exp_data/STEPS/{run_id}_000.csv',
                                usecols=['abs_seq', 'start', 'end'],
                                parse_dates=['start', 'end'])

        # frame data
        frames = pd.read_csv(f'./exp_data/FRAMES/{run_id}_000.csv',
                             parse_dates=['submitted', 'processed', 'returned'])

        # common metadata
        frames['run_id'] = run_id
        frames['square'] = sqr_mappings[run_id]
        frames['result'] = frames['result'].str.lower()

        # find initial frame
        init_frame = frames.loc[
            frames['result'].str.lower() == 'success', 'seq'
        ].values[0]

        init_frames.append({
            'run_id': run_id,
            # 'square': sqr_mappings[run_id],
            'frame' : int(init_frame)
        })

        for step in tqdm(run_steps.itertuples(name='Step'),
                         desc='Steps',
                         leave=False):
            # find all frames for step
            step_frames = frames.loc[
                (frames['submitted'] >= step.start) &
                (frames['returned'] <= step.end)
                ].copy()
            step_frames['step_seq'] = step.abs_seq

            # find the success frame
            success = step_frames.loc[
                (step_frames['returned'] == step.end) &
                (step_frames['result'].str.lower() == 'success')
                ]

            assert len(success) == 1
            success_frames.append(success)

    return pd.concat(success_frames, ignore_index=True), init_frames


@functools.cache
def load_frame(run_id: int, frame_seq: int) -> npt.NDArray:
    frame_path = out_dir / frame_path_fmt.format(
        run_id=run_id, frame=frame_seq
    )
    img = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
    return img


def tag_frame(frame: npt.NDArray,
              target_bitmap: npt.NDArray) -> str:
    target_state = BoardState(target_bitmap)
    try:
        board = BoardState(img_util.preprocess_img(frame))
        if board == target_state:
            return 'success'
        else:
            return 'error'
    except LowConfidenceError:
        return 'low_confidence'
    except LEGOCVError:
        return 'blank'


def imap_tag_frame(args: Tuple) -> str:
    return tag_frame(*args)


def whitewash_image(level: int,
                    image: npt.NDArray) -> npt.NDArray:
    max_arr_val = 255 - level
    transformed = image.copy()
    transformed[transformed > max_arr_val] = 255
    transformed[transformed <= max_arr_val] += level
    return transformed


# noinspection PyUnresolvedReferences
def generate_traces() -> None:
    success_frames, init_frames = find_frames()

    # fetch all frames
    remote_files = deque()

    # build list of frames to transfer
    for frame in init_frames:
        remote_files.append(
            frame_path_fmt.format(run_id=frame['run_id'],
                                  frame=frame['frame'])
        )

    for frame in success_frames.itertuples(index=True,
                                           name='Frame'):
        remote_files.append(
            frame_path_fmt.format(run_id=frame.run_id,
                                  frame=frame.seq))

    # transfer all frames
    print(f'RSyncing {len(remote_files)} files... ', end='')
    rsync = subprocess.Popen(
        ['rsync', '-a', remote_src,
         str(out_dir), f'--files-from=-'],
        encoding='utf8', stdin=subprocess.PIPE
    )
    rsync.communicate(
        '\n'.join(map(str, remote_files))
    )
    print('Done.')

    # find equivalent steps in latinsqrs
    # stores mapping from step state -> success frames
    states: Dict[str, pd.DataFrame] = defaultdict(pd.DataFrame)
    latin_squares = deque()

    for i in range(12):
        latin_sqr = pd.read_csv(f'./tasks/latin_sqr_{i:d}.csv')
        latin_squares.append(latin_sqr)
        for step in latin_sqr.itertuples(index=True, name='Step'):
            # find success frame for this square, step
            step_frames = success_frames.loc[
                (success_frames['step_seq'] == step.Index) &
                (success_frames['square'] == i)
                ]

            # concatenate frames into dict
            states[step.state] = pd.concat(
                (states[step.state], step_frames)
            )

    # for each step pattern, find frames which produce deterministic results
    frames_for_states = {}
    rng = np.random.default_rng()

    req_tags = {'success', 'low_confidence', 'blank'}

    for state_i, (board_bitmap_str, frames) in \
            tenumerate(states.items(), desc='Finding frames for states'):
        target_bitmap = np.array(eval(board_bitmap_str), dtype=np.uint8)
        results = {}
        with tqdm(total=len(req_tags), desc='Tags', leave=False) as res_pbar, \
                Pool() as pool:
            # find frames
            for frame in frames.sample(frac=1).itertuples():
                img_frame = load_frame(frame.run_id, frame.seq)
                frame_vars = pool.starmap(
                    whitewash_image,
                    [(i, img_frame) for i in
                     rng.choice(np.arange(1, 256, dtype=np.uint8),
                                size=255, replace=False)]
                )
                frame_vars = [img_frame] + frame_vars

                tags = pool.imap(
                    imap_tag_frame,
                    zip(frame_vars, itertools.repeat(target_bitmap))
                )

                done = False
                with tqdm(total=len(frame_vars),
                          desc=f'Finding frames for state {state_i}',
                          leave=False) as pbar:
                    for tag, frame in zip(tags, frame_vars):
                        pbar.update(1)
                        if tag not in results:
                            res_pbar.update(1)
                            results[tag] = frame
                            if req_tags <= set(results.keys()):
                                done = True
                                frames_for_states[
                                    board_bitmap_str] = results
                                break
                if done:
                    break

    # build traces for each latinsquare
    for square_i, square in tenumerate(latin_squares,
                                       desc='Building traces for squares'):
        init_frame_dict = rng.choice(init_frames)
        init_frame_img = load_frame(init_frame_dict['run_id'],
                                    init_frame_dict['frame'])
        frames_imgs = {}
        for step in tqdm(square.itertuples(index=True, name='Step'),
                         desc='Collecting steps', leave=False):
            for tag, img in frames_for_states[step.state].items():
                frames_imgs[f'step{step.Index:02d}_{tag}'] = img

        np.savez_compressed(
            f'./square{square_i:02d}.npz',
            initial=init_frame_img,
            **frames_imgs
        )


if __name__ == '__main__':
    generate_traces()
