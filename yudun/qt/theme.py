"""Ink-blue surfaces, jade interaction and a vermilion seal for 照微."""

WORKSTATION_STYLE = """
QMainWindow, QWidget#workspace { background: #e7edef; color: #213745; }
QWidget#masthead { background: #1b3041; border-bottom: 2px solid #b18d57; }
QLabel#seal {
    background: #a84135; border: 1px solid #d18573; border-radius: 6px;
    color: #fff1dd; font: bold 24px "Noto Serif CJK SC";
}
QLabel#wordmark { color: #faf3e8; font: bold 30px "Noto Serif CJK SC"; }
QLabel#mastheadTitle { color: #e4edf0; font-size: 15px; font-weight: 600; }
QLabel#inputCaption { color: #b8c8cc; font-size: 11px; }
QLabel#inputSource { color: #f3eee0; font-size: 13px; }
QGroupBox {
    background: #f9fbfa; border: 1px solid #cbd8d9; border-radius: 8px;
    margin-top: 10px; padding-top: 12px; color: #213745;
    font-size: 14px; font-weight: 700;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 12px; padding: 0 5px; background: #f9fbfa;
}
QGroupBox QLabel, QGroupBox QCheckBox { color: #304956; }
QLabel#cameraCanvas, QLabel#planCanvas {
    background: #142634; color: #d0dde0; border: 1px solid #243e4c; border-radius: 5px;
}
QLabel#caption { color: #526a76; font-size: 12px; }
QLabel#trackHeading { color: #234050; font-weight: 700; }
QLabel#timeReadout {
    color: #1b6157; background: #e4efed; padding: 5px 8px; border-radius: 4px;
    font-family: "DejaVu Sans Mono";
}
QPushButton, QComboBox {
    min-height: 22px; padding: 3px 8px; border: 1px solid #c1d2d5;
    border-radius: 5px; color: #243e4c; background: #f5f9f8;
}
QPushButton:hover, QComboBox:hover { background: #e4f0ee; border-color: #7ca9a2; }
QPushButton:pressed { background: #d1e5e1; }
QPushButton:focus, QComboBox:focus, QTabBar::tab:focus { border-color: #247568; }
QPushButton:disabled, QComboBox:disabled { color: #617582; background: #edf2f3; border-color: #d7e2e4; }
QPushButton#primary { background: #216d62; border-color: #216d62; color: #ffffff; font-weight: 700; }
QPushButton#primary:hover { background: #1b5e55; }
QPushButton#primary:focus { border: 1px solid #163748; }
QPushButton#primary:disabled { background: #e4ecec; color: #5c707b; border-color: #d0dddd; }
QTabWidget::pane { border: 1px solid #c6d4d7; background: #142634; }
QTabBar::tab {
    color: #455e68; background: #eaf0f0; border: 1px solid #c6d4d7; border-bottom: 0;
    border-top-left-radius: 4px; border-top-right-radius: 4px; padding: 6px 11px; margin-right: 3px;
}
QTabBar::tab:selected { color: #195c51; background: #f5faf8; border-top: 2px solid #267568; }
QTabBar::tab:hover { background: #e4f0ee; }
QTableWidget, QTextEdit {
    border: 1px solid #cbd8d9; border-radius: 4px; background: #ffffff; color: #253d4b;
    alternate-background-color: #edf4f3; selection-color: #183a34; selection-background-color: #cce4df;
}
QTableWidget { font-size: 12px; }
QTableWidget::item { padding: 3px 4px; }
QHeaderView::section {
    color: #405b65; background: #e7eff0; border: 0; padding: 5px 4px; font-weight: 700;
}
QSlider::groove:horizontal { height: 4px; border-radius: 2px; background: #d5e2e3; }
QSlider::sub-page:horizontal { background: #75aaa0; border-radius: 2px; }
QSlider::handle:horizontal { width: 12px; margin: -4px 0; border-radius: 6px; background: #246d63; }
QSlider::handle:horizontal:disabled { background: #9eafb6; }
QStatusBar { background: #dfe8e9; color: #3e5663; }
QStatusBar::item { border: 0; }
"""
