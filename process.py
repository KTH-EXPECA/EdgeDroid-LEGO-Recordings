import random
import tarfile
from collections import deque
from contextlib import closing
from dataclasses import dataclass
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Collection, Optional

import click
import pandas as pd
import yaml
from dataclasses_json import dataclass_json


def copy_frame_to_tar(
        frame_index: int,
        frame_tag: str,
        frame_dir: PathLike,
        tar_file: tarfile.TarFile,
        frame_step: Optional[int] = None
) -> None:
    frame_file = (Path(frame_dir) / f'frame_{frame_index:d}.jpeg').resolve()

    if frame_step is not None:
        tar_path = f'step_{frame_step:02d}/{frame_tag.lower()}.jpeg'
    else:
        tar_path = f'{frame_tag.lower()}.jpeg'

    tar_file.add(
        frame_file,
        arcname=tar_path
    )


@dataclass_json
@dataclass
class Metadata:
    task_name: str
    num_steps: int


@click.command()
@click.argument('output-file', type=click.Path(file_okay=True,
                                               dir_okay=False,
                                               exists=False))
@click.argument('num-steps', type=int)
@click.argument('frame-dirs', nargs=-1, type=click.Path(file_okay=False,
                                                        dir_okay=True,
                                                        exists=True))
@click.option('--task-name', type=str, default='latinsqr0', show_default=True)
def process_frames(output_file: PathLike,
                   num_steps: int,
                   frame_dirs: Collection[PathLike],
                   task_name: str) -> None:
    frame_options = deque()
    initial_frames = deque()
    for frame_dir in frame_dirs:
        # open csv file with frame data
        frame_dir = Path(frame_dir).resolve()
        frame_data = pd.read_csv(frame_dir / 'frames.csv',
                                 index_col='seq',
                                 dtype={'result': 'category'},
                                 infer_datetime_format=True,
                                 parse_dates=['submitted', 'processed',
                                              'returned'])

        step_data = pd.read_csv(frame_dir / 'steps.csv',
                                index_col='abs_seq',
                                infer_datetime_format=True,
                                parse_dates=['start', 'end'])

        init_frame = frame_data.loc[
            frame_data['result'].str.lower() == 'success'
            ].index[0]

        initial_frames.append((init_frame, frame_dir))

        for step in step_data.itertuples(name='Step', index=True):
            # find all frames in step
            step_frames = frame_data.loc[
                (frame_data['submitted'] >= step.start) &
                (frame_data['returned'] <= step.end)
                ].copy()

            # first find the corresponding success frame
            success_frame = step_frames.loc[
                frame_data['returned'] == step.end,
            ].index[0]

            frame_options.append({
                'tag'        : 'success',
                'frame_index': success_frame,
                'frame_dir'  : frame_dir,
                'step'       : step.Index
            })

            # rest of frames
            for frame in step_frames.loc[
                step_frames['result'].str.lower() != 'success'
            ].itertuples(name='Frame', index=True):
                frame_options.append({
                    'tag'        : frame.result.lower(),
                    'frame_index': frame.Index,
                    'frame_dir'  : frame_dir,
                    'step'       : step.Index
                })

    frames = pd.DataFrame(frame_options)

    with tarfile.open(output_file, 'w:gz') as tfile:
        # write metadata
        metadata = Metadata(
            task_name=task_name,
            num_steps=num_steps
        )

        mdata_bytes = yaml.safe_dump(metadata.to_dict()).encode('utf8')
        tinfo = tarfile.TarInfo(name='metadata.yml')
        tinfo.size = len(mdata_bytes)

        with closing(BytesIO(mdata_bytes)) as fp:
            tfile.addfile(tarinfo=tinfo, fileobj=fp)

        # write frames to tarfile
        # initial frame
        init_frame, frame_dir = random.choice(initial_frames)
        copy_frame_to_tar(
            frame_index=init_frame,
            frame_tag='initial',
            frame_dir=frame_dir,
            tar_file=tfile
        )

        # rest of frames
        for step in range(num_steps):
            # find corresponding frames
            step_frames = frames.loc[frames['step'] == step]

            # one frame per tag type
            for frame in step_frames.groupby('tag').sample(1) \
                    .itertuples(index=True):
                copy_frame_to_tar(
                    frame_index=frame.frame_index,
                    frame_tag=frame.tag,
                    frame_dir=frame.frame_dir,
                    frame_step=step,
                    tar_file=tfile
                )


if __name__ == '__main__':
    process_frames()
