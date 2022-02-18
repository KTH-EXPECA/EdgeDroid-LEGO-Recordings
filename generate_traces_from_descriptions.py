import subprocess
import tarfile
from collections import deque
from contextlib import closing
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import yaml

remote_src = 'blizzard@Blizzard:/home/blizzard/data/experiment_logs/'
frame_path_fmt = '{run_id}/main_task/frames/processed/frame_{frame:d}.jpeg'


def copy_frame_to_tar(
        run_id: int,
        frame_index: int,
        frame_tag: str,
        base_dir: Path,
        tar_file: tarfile.TarFile,
        frame_step: Optional[int] = None
) -> None:
    frame_file = base_dir / frame_path_fmt.format(run_id=run_id,
                                                  frame=frame_index)

    if frame_step is not None:
        tar_path = f'step_{frame_step:02d}/{frame_tag.lower()}.jpeg'
    else:
        tar_path = f'{frame_tag.lower()}.jpeg'

    tar_file.add(
        frame_file,
        arcname=tar_path
    )


if __name__ == '__main__':
    for desc in Path('./descriptions').glob('latin_square_*.yml'):
        with desc.open('r') as fp:
            description = yaml.safe_load(fp)
        square_num = int(desc.name[len('latin_square_'):-len('.yml')])
        print(f'Processing square {square_num}...')

        remote_files = deque()
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir).resolve()

            # build list of frames to transfer
            # initial frame
            init_run = description['initial']['run_id']
            init_frame = description['initial']['frame']
            remote_files.append(
                frame_path_fmt.format(run_id=init_run, frame=init_frame)
            )

            # steps
            for _, tags in description['steps'].items():
                for tag, specs in tags.items():
                    run_id = specs['run_id']
                    frame = specs['frame']

                    remote_files.append(
                        frame_path_fmt.format(run_id=run_id, frame=frame)
                    )

            # transfer all files to the temporary dir
            print('Transferring files... ', end='')
            rsync = subprocess.Popen(
                ['rsync', '-a', remote_src,
                 str(tmpdir), f'--files-from=-'],
                encoding='utf8', stdin=subprocess.PIPE
            )
            rsync.communicate(
                '\n'.join(map(str, remote_files))
            )
            print('Done.')

            # build the trace
            print('Building tarfile... ', end='')
            tfile_path = Path(f'./square{square_num:02d}.tgz')
            with tarfile.open(tfile_path, 'w:gz') as tfile:
                # write metadata
                metadata = dict(
                    task_name=description['task_name'],
                    num_steps=description['num_steps']
                )

                mdata_bytes = yaml.safe_dump(metadata).encode('utf8')
                tinfo = tarfile.TarInfo(name='metadata.yml')
                tinfo.size = len(mdata_bytes)

                with closing(BytesIO(mdata_bytes)) as fp:
                    tfile.addfile(tarinfo=tinfo, fileobj=fp)

                # write frames

                # init frame
                copy_frame_to_tar(
                    run_id=init_run,
                    frame_index=init_frame,
                    frame_tag='initial',
                    tar_file=tfile,
                    base_dir=tmpdir
                )

                for step, tags in description['steps'].items():
                    for tag, specs in tags.items():
                        run_id = specs['run_id']
                        frame = specs['frame']

                        copy_frame_to_tar(
                            run_id=run_id,
                            frame_index=frame,
                            frame_tag=tag,
                            frame_step=step,
                            tar_file=tfile,
                            base_dir=tmpdir
                        )

            print(f'done: {tfile_path}')
    print('All done!')
