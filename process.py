import tarfile
from contextlib import closing
from dataclasses import dataclass
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Optional

import click
import pandas as pd
import yaml
from dataclasses_json import dataclass_json


def copy_frame_to_tar(
        frame_index: int,
        frame_tag: str,
        frame_dir: Path,
        tar_file: tarfile.TarFile,
        frame_step: Optional[int] = None
) -> None:
    frame_file = (frame_dir / f'frame_{frame_index:d}.jpeg')

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
@click.argument('frame-dir', type=click.Path(file_okay=False,
                                             dir_okay=True,
                                             exists=True))
@click.argument('output-file', type=click.Path(file_okay=True,
                                               dir_okay=False,
                                               exists=False))
@click.option('--task-name', type=str, default='latinsqr0', show_default=True)
def process_frames(frame_dir: PathLike,
                   output_file: PathLike,
                   task_name: str) -> None:
    # open csv file with frame data
    frame_dir = Path(frame_dir).resolve()
    frame_data = pd.read_csv(frame_dir / 'frames.csv',
                             index_col='seq',
                             dtype={'result': 'category'},
                             infer_datetime_format=True,
                             parse_dates=['submitted', 'processed', 'returned'])
    step_data = pd.read_csv(frame_dir / 'steps.csv',
                            index_col='abs_seq',
                            infer_datetime_format=True,
                            parse_dates=['start', 'end'])

    with tarfile.open(output_file, 'w:gz') as tfile:

        # initial frame corresponds to first "success" in frame_data
        init_frame = frame_data.loc[
            frame_data['result'].str.lower() == 'success'
            ].index[0]

        copy_frame_to_tar(
            frame_index=init_frame,
            frame_tag='initial',
            frame_dir=frame_dir,
            tar_file=tfile
        )

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

            copy_frame_to_tar(
                frame_index=success_frame,
                frame_tag='success',
                frame_dir=frame_dir,
                frame_step=step.Index,
                tar_file=tfile
            )

            # sample the remaining categories
            step_frames = step_frames.loc[
                step_frames['result'].str.lower() != 'success'
                ].copy()
            step_frames['result'] = \
                step_frames['result'].cat.remove_unused_categories()

            sampled_frames = step_frames.groupby('result').sample(1)
            for frame in sampled_frames.itertuples(name='Frame', index=True):
                copy_frame_to_tar(
                    frame_index=frame.Index,
                    frame_tag=frame.result.lower(),
                    frame_dir=frame_dir,
                    frame_step=step.Index,
                    tar_file=tfile
                )

        # write metadata
        metadata = Metadata(
            task_name=task_name,
            num_steps=len(step_data.index)
        )

        mdata_bytes = yaml.safe_dump(metadata.to_dict()).encode('utf8')
        tinfo = tarfile.TarInfo(name='metadata.yml')
        tinfo.size = len(mdata_bytes)

        with closing(BytesIO(mdata_bytes)) as fp:
            tfile.addfile(tarinfo=tinfo, fileobj=fp)

        # done


if __name__ == '__main__':
    process_frames()
