import os
import subprocess
import json

INPUT = 'ip_test_chart.dng'
OUT = 'ip_test_chart_test'


def test_detect_grid_corners():
    # run detect_grid_corners
    cmd = ['python3', 'detect_grid_corners.py', INPUT, '--cols', '11', '--rows', '7', '--out', OUT, '--kernel-scale', '0.35']
    r = subprocess.run(cmd, capture_output=True, text=True)
    print('detect output:', r.stdout, r.stderr)
    assert r.returncode == 0
    assert os.path.exists(OUT + '.grid_debug.jpg')
    assert os.path.exists(OUT + '.grid_points.json')
    with open(OUT + '.grid_points.json') as f:
        js = json.load(f)
        assert 'corners' in js
        assert len(js['corners']) == 4


def test_rectify_raw_1d():
    # run rectify_raw_1d
    cmd = ['python3', 'rectify_raw_1d.py', INPUT, '--kernel-scale', '0.35']
    r = subprocess.run(cmd, capture_output=True, text=True)
    print('rectify output:', r.stdout, r.stderr)
    assert r.returncode == 0
    assert os.path.exists(INPUT + '.rectified_preview.jpg')
    assert os.path.exists(INPUT + '.rectified.tiff')


if __name__ == '__main__':
    test_detect_grid_corners()
    test_rectify_raw_1d()
    print('Tests done')
