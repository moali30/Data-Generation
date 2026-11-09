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
    parser.add_argument(
        "--self-test", action="store_true",
        help="run a headless end-to-end simulation and report, then exit; "
             "used to verify a packaged build before it is shipped")
    args = parser.parse_args(argv)

    if args.list_examples:
        from semsim.core.examples import EXAMPLES
        for name in EXAMPLES:
            print(name)
        return 0

    if args.self_test:
        return _self_test()

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


def _self_test() -> int:
    """Exercise the whole numerical stack and report, without a window.

    A packaged build can start, show its menus and list its examples while
    being quietly broken: PyInstaller's static analysis misses the scipy
    submodules imported lazily, so the failure surfaces the first time a
    p-value is computed -- which is after the researcher has specified a
    model and pressed Run. Checking that the interface opens proves almost
    nothing about that. This compiles a model, generates data, estimates it
    and computes the fit indices, so the build script can fail a bad bundle
    before it is shipped rather than after.
    """
    import traceback

    steps = []

    def step(name, fn):
        try:
            value = fn()
        except Exception:
            print(f"FAIL  {name}")
            traceback.print_exc()
            steps.append(False)
            return None
        print(f"ok    {name}: {value}")
        steps.append(True)
        return value

    print("SEM Data Simulation Engine - self test")
    import platform
    import sys as _sys
    frozen = getattr(_sys, "frozen", False)
    print(f"      python {platform.python_version()} on {platform.system()} "
          f"({'frozen bundle' if frozen else 'source checkout'})")

    from semsim.core import examples
    from semsim.core.compile import compile_model
    from semsim.core.validation import validate
    from semsim.generation.engine import generate_dataset

    spec = examples.simple_mediation()
    spec.power.n = 400
    spec.fit_target.enabled = False
    spec.noise.enabled = False

    compiled = step("compile", lambda: f"{spec.n_items} items, "
                                       f"df = {spec.degrees_of_freedom(compile_model(spec))}")
    step("validate", lambda: f"{len(validate(spec).errors())} error(s)")
    dataset = step("generate", lambda: generate_dataset(spec).data.shape)

    def estimate_and_fit():
        from semsim.analysis.estimator import estimate
        from semsim.analysis.fit_indices import compute
        data = generate_dataset(spec).data
        result = estimate(data, spec)
        fit = compute(result.sample_cov, result.sigma_hat, result.n,
                      result.n_free)
        return (f"RMSEA = {fit.rmsea:.4f}, CFI = {fit.cfi:.4f}, "
                f"chi2 = {fit.chi_square:.2f} on {fit.df} df")

    step("estimate + fit (exercises scipy)", estimate_and_fit)

    def reliability():
        from semsim.analysis.reliability import analyse
        report = analyse(generate_dataset(spec).data, spec)
        return ", ".join(f"{f.factor_name[:12]} alpha={f.cronbach_alpha:.3f}"
                         for f in report.per_factor)

    step("reliability", reliability)
    step("power (exercises scipy.stats.ncx2)", lambda: _power_line(spec))
    step("report", lambda: f"{len(_render_report(spec))} bytes of HTML")

    passed = sum(steps)
    print(f"\n{passed}/{len(steps)} checks passed")
    return 0 if passed == len(steps) else 1


def _power_line(spec) -> str:
    from semsim.core.compile import compile_model
    from semsim.power import analysis
    result = analysis.analyse(spec, compile_model(spec))
    return f"recommended N = {result.n_recommended}"


def _render_report(spec) -> str:
    from semsim.generation.engine import generate_dataset
    from semsim.io.report import ReportBundle, render_html
    return render_html(ReportBundle(spec=spec,
                                    dataset=generate_dataset(spec, n=120),
                                    generated_at="self-test"), lang="ar")


if __name__ == "__main__":
    raise SystemExit(main())
