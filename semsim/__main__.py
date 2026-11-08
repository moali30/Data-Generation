"""Entry point: ``python -m semsim`` or the ``semsim`` console script."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="semsim",
        description="SEM Data Simulation Engine - a virtual laboratory for "
                    "structural equation modelling data.")
    parser.add_argument("project", nargs="?",
                        help="a .semsim project file to open on startup")
    parser.add_argument("--lang", choices=["ar", "en"], default="ar",
                        help="interface language (default: ar)")
    parser.add_argument("--theme", choices=["light", "dark"], default="light")
    parser.add_argument("--example", help="load a bundled example by name")
    parser.add_argument("--list-examples", action="store_true")
    args = parser.parse_args(argv)

    if args.list_examples:
        from semsim.core.examples import EXAMPLES
        for name in EXAMPLES:
            print(name)
        return 0

    from semsim.ui import theme as theme_module
    from semsim.ui.i18n import translator
    translator.set_language(args.lang)

    from PySide6.QtWidgets import QApplication
    from semsim.ui.app import MainWindow, application_icon
    from semsim.ui.widgets.common import set_current_theme

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("SEM Data Simulation Engine")
    app.setOrganizationName("semsim")
    app.setWindowIcon(application_icon())
    set_current_theme(args.theme)
    theme_module.apply(app, args.theme, rtl=translator.is_rtl)

    window = MainWindow(theme_name=args.theme)
    if args.example:
        window.load_example(args.example)
    elif args.project:
        window.open_project(args.project)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
