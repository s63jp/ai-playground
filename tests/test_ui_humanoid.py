import subprocess, shutil, pathlib
import pytest

STATIC = pathlib.Path(__file__).parents[1] / 'playground' / 'dashboard' / 'static'

def test_scripts_present_and_check():
    # verify script tags in index.html
    html = (STATIC / 'index.html').read_text(encoding='utf-8')
    for src in ['/static/js/humanoid_view.js', '/static/js/splat.js', '/static/js/impact_fx.js', '/static/js/player.js']:
        assert f'<script type="module" src="{src}"' in html
    node = shutil.which('node')
    if not node:
        pytest.skip('node not available')
    for fn in ['app.js', 'js/humanoid_view.js', 'js/splat.js', 'js/impact_fx.js', 'js/player.js']:
        out = subprocess.run([node, '--check', str(STATIC / fn)], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
