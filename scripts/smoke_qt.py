"""Show a Linux Qt replay, capture it, and verify orderly window shutdown."""
import json
import sys
from pathlib import Path
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from yudun.pipeline import load_config
from yudun.qt.main import RadarStationMainWindow

if sys.platform != 'linux':
    raise SystemExit('Run inside Linux or WSL')
output = Path(__file__).resolve().parents[1]/'outputs/wsl-validation'
output.mkdir(parents=True, exist_ok=True)
app = QApplication([])
window = RadarStationMainWindow(load_config())
window.show()
result = {'platform': app.platformName(), 'frames': 0, 'closed': False}

def capture():
    if not window.frames:
        return
    timer.stop()
    window.render(window.frames[len(window.frames)//2])
    app.processEvents()
    assert window.grab().save(str(output/'qt-wslg.png'))
    result['frames'] = len(window.frames)
    result['mode_count'] = window.mode.count()
    result['has_lidar_status'] = hasattr(window, 'lidar_status')
    result['closed'] = window.close()
    (output/'qt-wslg.json').write_text(json.dumps(result, indent=2)+'\n')
    app.quit()

timer = QTimer()
timer.timeout.connect(capture)
timer.start(100)
QTimer.singleShot(15000, app.quit)
app.exec_()
assert result['frames'] and result['closed'] and result['mode_count'] == 2
assert not result['has_lidar_status']
print(json.dumps(result, indent=2))
