# Copyright (c) 2024 -      Dana Diaconu
# Copyright (c) 2017 - 2020 Kensho Hara

# MIT License
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import sys
import json
import subprocess
from pathlib import Path

if __name__ == '__main__':
    video_dir_path = Path(sys.argv[1])
    json_path = Path(sys.argv[2])
    if len(sys.argv) > 3:
        dst_json_path = Path(sys.argv[3])
    else:
        dst_json_path = json_path

    with json_path.open('r') as f:
        json_data = json.load(f)

    for video_file_path in sorted(video_dir_path.iterdir()):
        file_name = video_file_path.name
        if '.mp4' not in file_name:
            continue
        name = video_file_path.stem

        ffprobe_cmd = ['ffprobe', str(video_file_path)]
        p = subprocess.Popen(
            ffprobe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res = p.communicate()[1].decode('utf-8')

        fps = float([x for x in res.split(',') if 'fps' in x][0].rstrip('fps'))
        json_data['database'][name[2:]]['fps'] = fps

    with dst_json_path.open('w') as f:
        json.dump(json_data, f)