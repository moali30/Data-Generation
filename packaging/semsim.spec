# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the SEM Data Simulation Engine (Windows).

Build (from anywhere; paths resolve against this file's location):

    pip install pyinstaller
    pyinstaller packaging/semsim.spec --noconfirm

Produces ``dist/SEMSimulator.exe`` -- one file, no console window, no Python
installation required on the target machine.

Notes that matter for this particular application
-------------------------------------------------
*   **scipy.** PyInstaller's static analysis misses several scipy submodules
    that are imported lazily (``scipy.special.cython_special``,
    ``scipy._lib.messagestream``, the ``scipy.stats`` distribution machinery).
    They are listed explicitly below; without them the estimator raises an
    ImportError the first time a p-value is computed, which is well after the
    user has started a run.
*   **Qt plugins.** Only the platform, style and image-format plugins are
    needed. The Qt WebEngine, Quick, 3D, Multimedia and SQL drivers are large
    and unused, so they are excluded; that is most of the difference between a
    ~90 MB build and a ~250 MB one.
*   **matplotlib** is a dependency of nothing on the critical path -- every
    chart in the interface and in the report is drawn with QPainter or emitted
    as SVG -- so it is excluded entirely.
"""

import os
import sys

BLOCK_CIPHER = None

# PyInstaller resolves a relative path in a .spec against the SPEC FILE's
# directory, not the working directory, so deriving the repository root from
# os.getcwd() breaks the moment the build is launched from anywhere but the
# root -- it went looking for packaging/semsim/__main__.py. SPECPATH is the
# directory holding this file, injected by PyInstaller, so the root is its
# parent and the build works from any working directory.
ROOT = os.path.dirname(os.path.abspath(SPECPATH))  # noqa: F821 - injected

hidden = [
    # scipy pieces imported lazily at first use
    "scipy.special.cython_special",
    "scipy._lib.messagestream",
    "scipy._cyutility",
    "scipy.sparse.csgraph._validation",
    "scipy.optimize._minimize",
    "numpy.testing",
    "scipy.optimize._linesearch",
    "scipy.stats._continuous_distns",
    "scipy.stats._discrete_distns",
    "scipy.stats._multivariate",
    # SPSS export
    "pyreadstat",
    "pyreadstat._readstat_writer",
    # spreadsheet export
    "openpyxl.cell._writer",
]

# The interface pages are imported by name at run time (PAGE_SPECS in
# semsim/ui/app.py), so static analysis cannot find them and they must be listed
# here. Walk the source tree instead of calling collect_submodules("semsim"):
# that needs semsim to be importable while this spec runs, and it is not
# installed then, so it returned an empty list without any error and v1.0.0
# shipped with every page replaced by a ModuleNotFoundError placeholder.
for folder, subfolders, files in os.walk(os.path.join(ROOT, "semsim")):
    subfolders[:] = [name for name in subfolders if name != "__pycache__"]
    package = os.path.relpath(folder, ROOT).replace(os.sep, ".")
    for name in files:
        if name.endswith(".py"):
            hidden.append(package if name == "__init__.py"
                          else f"{package}.{name[:-3]}")

datas = []
resources = os.path.join(ROOT, "semsim", "resources")
if os.path.isdir(resources) and os.listdir(resources):
    datas.append((resources, "semsim/resources"))

# Excluding a stdlib module to shave megabytes is a false economy when a
# scientific dependency imports it lazily. "unittest" was on this list, and
# numpy.testing imports it at module scope: the bundle started, showed its
# menus and listed its examples, then raised ModuleNotFoundError the moment
# the user pressed Run. The --self-test entry point exists to catch exactly
# that, and did. Only genuinely unused third-party packages are excluded now.
excludes = [
    "matplotlib", "tkinter", "IPython", "jupyter", "notebook", "pytest",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick", "PySide6.QtQuick", "PySide6.QtQuick3D",
    "PySide6.QtQml", "PySide6.Qt3DCore", "PySide6.Qt3DRender",
    "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets",
    "PySide6.QtBluetooth", "PySide6.QtNfc", "PySide6.QtPositioning",
    "PySide6.QtSerialPort", "PySide6.QtSql", "PySide6.QtTest",
    "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtCharts",
    "PySide6.QtDataVisualization", "PySide6.QtNetworkAuth",
    "PySide6.QtRemoteObjects", "PySide6.QtScxml", "PySide6.QtSensors",
    "PySide6.QtSpatialAudio", "PySide6.QtTextToSpeech", "PySide6.QtWebSockets",
    "PySide6.QtWebChannel", "PySide6.QtPdf", "PySide6.QtPdfWidgets",
]

analysis = Analysis(
    [os.path.join(ROOT, "semsim", "__main__.py")],
    pathex=[ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    cipher=BLOCK_CIPHER,
)

pyz = PYZ(analysis.pure, analysis.zipped_data, cipher=BLOCK_CIPHER)

icon_path = os.path.join(ROOT, "packaging", "semsim.ico")
icon = icon_path if os.path.exists(icon_path) else None

exe = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.zipfiles,
    analysis.datas,
    [],
    name="SEMSimulator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,              # UPX corrupts some Qt DLLs; the size win is not worth it
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # a GUI application must not flash a console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon,
    version=os.path.join(ROOT, "packaging", "version_info.txt")
    if os.path.exists(os.path.join(ROOT, "packaging", "version_info.txt"))
    else None,
)
