from collections import defaultdict
import multiprocessing
import os
import shutil
import subprocess

def createAnimation(
    files: list, output: list | str, 
    fps: int = None):
    if not isinstance(output, list):
        output = [output]
    
    prefix = multiprocessing.current_process().pid
    dirName = os.path.dirname(files[0])
    suffix = os.path.basename(files[0]).rsplit('.', 1)[-1]
    cmd = ['ffmpeg', '-y', '-pattern_type', 'sequence']
    if fps:
        cmd.append('-framerate')
        cmd.append(f'{fps}')
    cmd += ['-i', f'{dirName}/{prefix}.%06d.{suffix}']
    if fps:
        cmd.append('-r')
        cmd.append(f'{fps}')
    cmd += ['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p']
    repeatInputMap = defaultdict(list)
    for i, img in enumerate(files):
        repeatInputMap[img].append(i)
    for i, img in enumerate(files):
        dirName = os.path.dirname(img)
        if repeatInputMap[img][0] == i:
            os.symlink(img, f'{dirName}/{prefix}.{i:06d}.{suffix}')
        else:
            orgIdx = repeatInputMap[img][0]
            shutil.copyfile(
                f'{dirName}/{prefix}.{orgIdx:06d}.{suffix}', 
                f'{dirName}/{prefix}.{i:06d}.{suffix}')
    for i in output:
        iCmd = cmd + [i]
        subprocess.run(iCmd)
    for i, img in enumerate(files):
        os.remove(f'{dirName}/{prefix}.{i:06d}.{suffix}')