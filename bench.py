#!/usr/bin/env python3

import os
from os import path


threads = [1, 2, 4, 8]
images = ['pillars', 'lenna', 'lion', 'bungalows', 'sand_dunes', 'pinnacles', 'new_zealand']
images.remove('lion')

for img in images:
    img_path = path.join('images', img) + '.jpg'
    out_path = img
    os.makedirs(out_path, exist_ok=True)
    for t in threads:
        # :notlikeblob:
        os.system('./test-omp -t {0} -o {1} | tee {2}.txt'.format(t, out_path, path.join(out_path, 't{}'.format(t))))
