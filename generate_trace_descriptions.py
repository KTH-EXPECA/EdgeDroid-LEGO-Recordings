from collections import defaultdict, deque
from typing import Dict

import numpy as np
import pandas as pd
import yaml

if __name__ == '__main__':
    step_independent_tags = ('blank', 'low_confidence')

    sqr_mappings = pd.read_csv('latinsqr_mappings.csv',
                               index_col='run_id')
    sqr_mappings = sqr_mappings['latin_sqr_nr'].to_dict()

    success_frames = deque()
    initial_frames = deque()
    ind_frames = deque()
    for run_id in sqr_mappings.keys():
        # TODO: initial

        # step data
        run_steps = pd.read_csv(f'./exp_data/STEPS/{run_id}_000.csv',
                                usecols=['abs_seq', 'start', 'end'],
                                parse_dates=['start', 'end'])

        # frame data
        frames = pd.read_csv(f'./exp_data/FRAMES/{run_id}_000.csv',
                             parse_dates=['submitted', 'processed', 'returned'])
        frames['run_id'] = run_id
        frames['square'] = sqr_mappings[run_id]
        frames['result'] = frames['result'].str.lower()

        # remove task errors and no change
        frames = frames.loc[
            ~np.isin(frames['result'].str.lower(), ('task_error', 'no_change'))
        ].copy()

        # convert junk_frame and cv_error to blank_frame
        frames['result'] = frames['result'].replace({
            'junk_frame': 'blank', 'cv_error': 'blank'
        })

        # find initial frame
        init_frame = frames.loc[
            frames['result'].str.lower() == 'success', 'seq'
        ].values[0]

        initial_frames.append({
            'run_id': run_id,
            # 'square': sqr_mappings[run_id],
            'frame' : int(init_frame)
        })

        # save state independent frames
        ind_frames.append(
            frames.loc[
                np.isin(frames['result'].str.lower(), step_independent_tags)
            ].copy()
        )

        for step in run_steps.itertuples(name='Step'):
            # find the success frame of each step

            # find all frames in each step
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

            # all other frames
            # other: pd.DataFrame = step_frames.loc[
            #     step_frames['result'].str.lower() != 'success'
            #     ]

            # df = step_frames.loc[other.index.union(success.index)].copy()
            # df = df[['run_id', 'step_seq', 'seq', 'result', 'square']]
            #
            # frame_data.append(df)

    success_frames = pd.concat(success_frames, ignore_index=True)
    ind_frames = pd.concat(ind_frames, ignore_index=True)

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

    for i, square in enumerate(latin_squares):
        # finally, actually build the descriptions
        trace = {
            'task_name': f'latin_square_{i:02d}',
            'steps'    : {},
            'initial'  : np.random.choice(initial_frames)
        }

        for step in square.itertuples(index=True, name='Step'):
            # find possible success frames for step
            step_success_frames = states[step.state]

            # for state independent tags we sample the whole collection
            frames = pd.concat((step_success_frames, ind_frames),
                               ignore_index=True)

            # sample frames
            sampled_frames = frames.groupby('result').sample(1)

            trace['steps'][step.Index] = {
                frame.result.lower(): {
                    'run_id': frame.run_id,
                    'frame' : frame.seq
                }
                for frame in sampled_frames.itertuples(index=True, name='Frame')
            }

        trace['num_steps'] = len(trace['steps'])

        # verify trace
        for step in range(trace['num_steps']):
            for tag in ['success'] + list(step_independent_tags):
                assert tag in trace['steps'][step]

        # output trace
        with open(f'descriptions/latin_square_{i:02d}.yml', 'w') as fp:
            yaml.safe_dump(trace, fp)
