"""
semsim.ui.i18n
==============
The bilingual string catalogue and the live language switch.

Design
------
*   One flat ``dict[key] -> (arabic, english)`` table.  Flat beats nested here:
    a missing key is immediately obvious, and the table can be diffed.
*   :func:`tr` never raises.  An unknown key comes back as ``"?key"`` so a typo
    shows up in the interface instead of crashing the dialog that contains it.
*   :class:`Translator` emits ``language_changed`` so every open widget can
    re-label itself without the window being rebuilt.
*   Arabic terminology follows the usage of published Arabic methodology texts:
    the Arabic term first, with the English technical term in parentheses the
    first time it appears in a screen.  Researchers read and cite both.
*   Numbers stay in Western Arabic numerals (0-9) in both languages, because
    that is what SPSS, AMOS and every journal table use.
"""

from __future__ import annotations

import re

from typing import Dict, Iterable, List, Optional, Tuple

try:  # the catalogue is useful headless (reports, CLI) as well
    from PySide6.QtCore import QObject, Signal
    _HAVE_QT = True
except Exception:  # pragma: no cover - headless fallback
    _HAVE_QT = False

    class QObject:  # type: ignore[no-redef]
        def __init__(self, *a, **k) -> None:
            pass

    class Signal:  # type: ignore[no-redef]
        def __init__(self, *a, **k) -> None:
            self._subs: List = []

        def __get__(self, obj, objtype=None):
            return self

        def connect(self, fn) -> None:
            self._subs.append(fn)

        def emit(self, *a) -> None:
            for fn in list(self._subs):
                fn(*a)


ARABIC = "ar"
ENGLISH = "en"
LANGUAGES: Tuple[str, str] = (ARABIC, ENGLISH)

LANGUAGE_NAMES = {ARABIC: "العربية", ENGLISH: "English"}


# --------------------------------------------------------------------------- #
# The catalogue:  key -> (Arabic, English)
# --------------------------------------------------------------------------- #

STRINGS: Dict[str, Tuple[str, str]] = {

    # ---- application chrome ------------------------------------------- #
    "app.title": ("محرك محاكاة بيانات النمذجة بالمعادلات البنائية",
                  "SEM Data Simulation Engine"),
    "app.short_title": ("محاكي بيانات SEM", "SEM Simulator"),
    "app.tagline": (
        "مختبر افتراضي لتوليد بيانات تحاكي استجابات البشر الحقيقية",
        "A virtual laboratory for data that behaves like real human responses"),

    # ---- menus -------------------------------------------------------- #
    "menu.file": ("ملف", "File"),
    "menu.file.new": ("مشروع جديد", "New project"),
    "menu.file.open": ("فتح مشروع…", "Open project…"),
    "menu.file.save": ("حفظ", "Save"),
    "menu.file.save_as": ("حفظ باسم…", "Save as…"),
    "menu.file.recent": ("المشاريع الأخيرة", "Recent projects"),
    "menu.file.export": ("تصدير البيانات…", "Export data…"),
    "menu.file.exit": ("إغلاق", "Exit"),
    "menu.model": ("النموذج", "Model"),
    "menu.model.validate": ("فحص النموذج", "Validate model"),
    "menu.model.example": ("تحميل نموذج مثال", "Load example model"),
    "menu.model.clear": ("تفريغ النموذج", "Clear model"),
    "menu.run": ("تشغيل", "Run"),
    "menu.run.simulate": ("توليد البيانات", "Generate data"),
    "menu.run.estimate": ("تقدير النموذج", "Estimate model"),
    "menu.run.full": ("محاكاة ومعايرة كاملة", "Full simulation and calibration"),
    "menu.run.cancel": ("إيقاف", "Stop"),
    "menu.view": ("عرض", "View"),
    "menu.view.language": ("اللغة", "Language"),
    "menu.view.theme": ("المظهر", "Theme"),
    "menu.view.theme.light": ("فاتح", "Light"),
    "menu.view.theme.dark": ("داكن", "Dark"),
    "menu.help": ("مساعدة", "Help"),
    "menu.help.guide": ("دليل الاستخدام", "User guide"),
    "menu.help.methods": ("المراجع المنهجية", "Methodological references"),
    "menu.help.about": ("عن البرنامج", "About"),

    # ---- the six stages ----------------------------------------------- #
    "stage.1.title": ("بناء مواصفات النموذج", "Model specification"),
    "stage.1.subtitle": ("Model Specification Architecture",
                         "Define constructs, dimensions and items"),
    "stage.2.title": ("محاكاة تعقيدات الواقع", "Complex realities"),
    "stage.2.subtitle": ("Complex Realities Simulation",
                         "Cross-loadings, correlated errors, method bias"),
    "stage.3.title": ("توليد البيانات والتوزيعات", "Distributions and generation"),
    "stage.3.subtitle": ("Data Distribution & Generation Engine",
                         "Likert thresholds, skewness, kurtosis, noise"),
    "stage.4.title": ("تحليل القوة وحجم العينة", "Power and sample size"),
    "stage.4.subtitle": ("Power Analysis & Sample Size",
                         "Determine the N your model actually needs"),
    "stage.5.title": ("معايرة الدلالة وجودة التوفيق",
                      "Significance and fit targeting"),
    "stage.5.subtitle": ("Fit Indices & Significance Targeting",
                         "Aim each path and the global fit at a chosen outcome"),
    "stage.6.title": ("التقييم واستخراج المؤشرات", "Validation and output"),
    "stage.6.subtitle": ("Validation & Fit Output",
                         "Reliability, validity and model fit"),

    # ---- stage 1: measurement model ----------------------------------- #
    "m.constructs": ("المتغيرات الكامنة", "Latent variables"),
    "m.construct": ("المتغير الكامن", "Latent variable"),
    "m.add_construct": ("إضافة متغير كامن", "Add latent variable"),
    "m.add_second_order": ("إضافة متغير من الدرجة الثانية",
                           "Add second-order construct"),
    "m.duplicate": ("تكرار", "Duplicate"),
    "m.remove": ("حذف", "Remove"),
    "m.move_up": ("تحريك لأعلى", "Move up"),
    "m.move_down": ("تحريك لأسفل", "Move down"),
    "m.name": ("الاسم المختصر", "Short name"),
    "m.name.hint": ("يُستخدم كاسم العمود في البيانات المصدَّرة (حروف إنجليزية)",
                    "Used as the column name in the exported dataset"),
    "m.label_ar": ("التسمية بالعربية", "Arabic label"),
    "m.label_en": ("التسمية بالإنجليزية", "English label"),
    "m.role": ("الدور في النموذج", "Role in the model"),
    "m.order": ("رتبة المتغير", "Construct order"),
    "m.order.first": ("الدرجة الأولى — الأسئلة تُحمَّل عليه مباشرة",
                      "First-order — items load directly"),
    "m.order.second": ("الدرجة الثانية — يتكوّن من أبعاد",
                       "Second-order — built from dimensions"),
    "m.dimensions": ("الأبعاد", "Dimensions"),
    "m.dimension": ("البعد", "Dimension"),
    "m.add_dimension": ("إضافة بُعد", "Add dimension"),
    "m.n_dimensions": ("عدد الأبعاد", "Number of dimensions"),
    "m.items": ("الأسئلة (المؤشرات)", "Items (indicators)"),
    "m.item": ("السؤال", "Item"),
    "m.add_item": ("إضافة سؤال", "Add item"),
    "m.n_items": ("عدد الأسئلة", "Number of items"),
    "m.items_per_dimension": ("عدد الأسئلة لكل بُعد", "Items per dimension"),
    "m.loading": ("التشبع", "Loading"),
    "m.loading.full": ("التشبع المعياري (Standardized Loading)",
                       "Standardised loading"),
    "m.loading.hint": (
        "قيمة من 0 إلى 1. المعيار المقبول ≥ 0.70؛ وما دون 0.50 يُحذف عادةً.",
        "0 to 1. Convention: ≥ 0.70 is good, below 0.50 is usually dropped."),
    "m.second_order_loading": ("تشبع البعد على المتغير",
                               "Loading on the second-order construct"),
    "m.reverse_coded": ("سؤال معكوس الصياغة", "Reverse-worded item"),
    "m.extra_error": ("خطأ قياس إضافي", "Extra measurement error"),
    "m.uniform_loadings": ("توحيد التشبعات", "Set all loadings"),
    "m.vary_loadings": ("تنويع التشبعات عشوائيًا", "Vary loadings randomly"),

    "role.independent": ("مستقل", "Independent"),
    "role.dependent": ("تابع", "Dependent"),
    "role.mediator": ("وسيط", "Mediator"),
    "role.moderator": ("معدِّل", "Moderator"),
    "role.control": ("ضابط", "Control"),

    # ---- stage 1: structural model ------------------------------------ #
    "s.structural": ("النموذج الهيكلي", "Structural model"),
    "s.paths": ("المسارات", "Paths"),
    "s.path": ("المسار", "Path"),
    "s.add_path": ("إضافة مسار", "Add path"),
    "s.source": ("المصدر", "From"),
    "s.target": ("الهدف", "To"),
    "s.effect_size": ("قوة التأثير", "Effect size"),
    "s.coefficient": ("معامل المسار", "Path coefficient"),
    "s.coefficient.full": ("المعامل المعياري β", "Standardised coefficient β"),
    "s.significant": ("مستهدف أن يكون دالًا", "Target as significant"),
    "s.not_significant": ("مستهدف أن يكون غير دال", "Target as non-significant"),
    "s.latent_cov": ("التغاير بين المتغيرات المستقلة",
                     "Covariances among exogenous constructs"),
    "s.add_latent_cov": ("إضافة تغاير", "Add covariance"),
    "s.correlation": ("الارتباط", "Correlation"),
    "s.interactions": ("التأثيرات التفاعلية (التعديل)",
                       "Interaction (moderation) effects"),
    "s.add_interaction": ("إضافة تفاعل", "Add interaction"),
    "s.predictor": ("المتغير المستقل", "Predictor"),
    "s.moderator": ("المتغير المعدِّل", "Moderator"),
    "s.indirect": ("التأثيرات غير المباشرة (الوساطة)",
                   "Indirect (mediation) effects"),
    "s.add_indirect": ("إضافة مسار وساطة", "Add mediation chain"),
    "s.chain": ("سلسلة الوساطة", "Mediation chain"),
    "s.diagram": ("مخطط المسار", "Path diagram"),
    "s.auto_layout": ("ترتيب تلقائي", "Auto layout"),
    "s.diagram.show_items": ("إظهار الأسئلة", "Show items"),
    "s.diagram.mirror": ("اتجاه المخطط من اليمين لليسار",
                         "Right-to-left diagram flow"),
    "s.diagram.fit": ("ملء الشاشة", "Fit to view"),
    "s.diagram.export": ("تصدير المخطط صورة", "Export diagram as image"),
    "s.diagram.empty": ("لا يوجد متغيرات كامنة لعرضها بعد",
                        "No latent variables to draw yet"),

    "effect.none": ("لا يوجد تأثير", "None"),
    "effect.trivial": ("ضئيل جدًا", "Trivial"),
    "effect.weak": ("ضعيف", "Weak"),
    "effect.medium": ("متوسط", "Medium"),
    "effect.strong": ("قوي", "Strong"),
    "effect.very_strong": ("قوي جدًا", "Very strong"),
    "effect.custom": ("مخصّص", "Custom"),

    # ---- stage 2: complex realities ----------------------------------- #
    "x.cross_loadings": ("التأثيرات المتقاطعة", "Cross-loadings"),
    "x.cross_loadings.explain": (
        "تحميل سؤال واحد على أكثر من متغير كامن — يحاكي الواقع حيث قد يقيس "
        "السؤال أكثر من سمة، ويتيح دراسة أثر ذلك على الصدق التمايزي.",
        "One item loading on more than one construct. This mimics real "
        "questionnaires, where a single question can tap more than one trait, "
        "and lets you study the effect on discriminant validity."),
    "x.add_cross_loading": ("إضافة تحميل متقاطع", "Add cross-loading"),
    "x.on_factor": ("على المتغير", "On construct"),
    "x.auto_policy": ("توليد تلقائي للتحميلات المتقاطعة",
                      "Auto-generate cross-loadings"),
    "x.proportion_items": ("نسبة الأسئلة المتأثرة", "Share of affected items"),
    "x.magnitude_range": ("مدى الحجم", "Magnitude range"),
    "x.across_only": ("بين المتغيرات المختلفة فقط",
                      "Across different constructs only"),
    "x.allow_negative": ("السماح بتحميلات سالبة", "Allow negative loadings"),
    "x.max_per_item": ("أقصى عدد لكل سؤال", "Maximum per item"),

    "x.error_cov": ("التغاير بين الأخطاء", "Error covariances"),
    "x.error_cov.explain": (
        "ربط أخطاء القياس بين أسئلة محددة — يحاكي أثر طريقة القياس المشتركة "
        "(Common Method Bias) أو تشابه صياغة الأسئلة.",
        "Linking the measurement errors of specific items. This reproduces "
        "common method bias, or simply two questions worded alike."),
    "x.add_error_cov": ("إضافة تغاير بين خطأين", "Add error covariance"),
    "x.item_a": ("السؤال الأول", "First item"),
    "x.item_b": ("السؤال الثاني", "Second item"),
    "x.reason": ("السبب", "Reason"),
    "x.cmb": ("عامل طريقة القياس المشترك", "Common method factor"),
    "x.cmb.explain": (
        "عامل كامن غير مقيس يُحمَّل على كل الأسئلة، يحاكي تحيّز الأداة الواحدة "
        "والمصدر الواحد في الاستبيانات الذاتية.",
        "An unmeasured latent factor loading on every item: the single-source, "
        "single-instrument bias of self-report questionnaires."),
    "x.cmb_loading": ("تشبع عامل الطريقة", "Method factor loading"),
    "x.cmb_all_items": ("يشمل كل الأسئلة", "Applies to all items"),

    # ---- stage 3: distributions --------------------------------------- #
    "d.scale": ("مقياس الاستجابة", "Response scale"),
    "d.scale_kind": ("نوع المقياس", "Scale type"),
    "d.scale.continuous": ("متصل", "Continuous"),
    "d.scale.likert": ("ليكرت (فئوي مرتب)", "Likert (ordinal)"),
    "d.n_categories": ("عدد الفئات", "Number of categories"),
    "d.response_style": ("شكل توزيع الاستجابة", "Response distribution shape"),
    "d.thresholds": ("العتبات", "Thresholds"),
    "d.thresholds.explain": (
        "يستخدم البرنامج نماذج العتبات (Threshold Models) لتحويل المتغير الكامن "
        "المتصل إلى استجابة فئوية، بدلًا من التقريب البسيط الذي يشوّه التوزيع.",
        "The engine uses threshold models to turn the continuous latent "
        "response into a category, instead of naive rounding, which distorts "
        "the distribution."),
    "d.proportions": ("نسب الفئات", "Category proportions"),
    "d.min_label": ("تسمية أدنى فئة", "Lowest category label"),
    "d.max_label": ("تسمية أعلى فئة", "Highest category label"),
    "d.mean": ("المتوسط", "Mean"),
    "d.sd": ("الانحراف المعياري", "Standard deviation"),

    "style.symmetric": ("متماثل (طبيعي)", "Symmetric (normal)"),
    "style.uniform": ("منتظم", "Uniform"),
    "style.positive_skew": ("التواء موجب (تجمّع في الفئات المنخفضة)",
                            "Positive skew (floor effect)"),
    "style.negative_skew": ("التواء سالب (تجمّع في الفئات المرتفعة)",
                            "Negative skew (ceiling effect)"),
    "style.extreme_positive": ("التواء موجب شديد", "Extreme positive skew"),
    "style.extreme_negative": ("التواء سالب شديد", "Extreme negative skew"),
    "style.bimodal": ("ثنائي المنوال (رأي مستقطب)", "Bimodal (polarised)"),
    "style.custom": ("مخصّص", "Custom"),

    "d.nonnormality": ("الالتواء والتفلطح", "Skewness and kurtosis"),
    "d.nonnormality.explain": (
        "تحكّم كامل في شكل التوزيع ليطابق الظواهر الواقعية — مثل التواء موجب "
        "قوي لمحاكاة بيانات الدخل، أو التواء سالب لبيانات الأداء — ممّا يسمح "
        "باختبار النموذج في ظل عدم تحقق الاعتدالية.",
        "Full control of distribution shape so it matches reality: strong "
        "positive skew for income, negative skew for performance ratings. This "
        "is how you test a model under non-normality."),
    "d.skewness": ("الالتواء", "Skewness"),
    "d.kurtosis": ("التفلطح", "Kurtosis"),
    "d.kurtosis.hint": ("تفلطح زائد: 0 = توزيع طبيعي",
                        "Excess kurtosis: 0 means normal"),
    "d.use_nonnormal": ("تشغيل محرك التوزيعات غير الطبيعية (Vale-Maurelli)",
                        "Enable the non-normal engine (Vale-Maurelli)"),
    "d.default_skew": ("الالتواء الافتراضي لكل الأسئلة",
                       "Default skewness for all items"),
    "d.default_kurt": ("التفلطح الافتراضي لكل الأسئلة",
                       "Default kurtosis for all items"),
    "d.infeasible": (
        "تركيبة الالتواء والتفلطح غير قابلة للتحقيق رياضيًا — يجب أن يكون "
        "التفلطح أكبر من (الالتواء² − 2).",
        "This skewness/kurtosis pair is mathematically impossible: kurtosis "
        "must exceed skewness² − 2."),

    "d.noise": ("حقن التشويش العشوائي", "Noise injection"),
    "d.noise.explain": (
        "إضافة تباين غير مفسَّر لتجنّب المطابقة المفرطة (Overfitting)، لتظهر "
        "البيانات وكأنها جُمعت من عينة بشرية حقيقية.",
        "Unexplained variance that keeps the data from fitting too perfectly, "
        "so it reads like a real human sample."),
    "d.noise.enable": ("تشغيل التشويش", "Enable noise"),
    "d.unique_inflation": ("تضخيم تباين الخطأ", "Unique-variance inflation"),
    "d.unique_inflation.hint": (
        "1.0 = بلا تغيير. القيم الأعلى تُضعف الثبات كما يحدث في القياس الواقعي.",
        "1.0 leaves it alone. Higher values weaken reliability the way real "
        "measurement does."),
    "d.careless": ("نسبة المستجيبين غير المبالين", "Careless respondents"),
    "d.careless.explain": (
        "مستجيبون يجيبون دون قراءة — إمّا بخط مستقيم (نفس الإجابة لكل الأسئلة) "
        "أو بإجابات عشوائية. ظاهرة حقيقية في الاستبيانات الإلكترونية.",
        "Respondents who answer without reading: a straight line of identical "
        "answers, or random clicking. A real feature of online surveys."),
    "d.careless_style": ("نمط عدم المبالاة", "Careless style"),
    "d.careless.random": ("عشوائي", "Random"),
    "d.careless.straightline": ("خط مستقيم", "Straight-lining"),
    "d.careless.mixed": ("مختلط", "Mixed"),
    "d.outliers": ("نسبة القيم الشاذة", "Outliers"),
    "d.outlier_magnitude": ("حجم الشذوذ (بالانحرافات المعيارية)",
                            "Outlier magnitude (SD)"),
    "d.missing": ("البيانات المفقودة", "Missing data"),
    "d.missing_mechanism": ("آلية الفقد", "Missingness mechanism"),
    "d.missing.none": ("لا يوجد فقد", "None"),
    "d.missing.mcar": ("فقد عشوائي تمامًا (MCAR)",
                       "Missing completely at random (MCAR)"),
    "d.missing.mar": ("فقد عشوائي مشروط (MAR)", "Missing at random (MAR)"),
    "d.missing_proportion": ("نسبة الفقد", "Missing proportion"),
    "d.mar_driver": ("المتغير المسبّب للفقد", "Driver of missingness"),
    "d.jitter": ("اهتزاز القياس", "Measurement jitter"),
    "d.seed": ("بذرة العشوائية", "Random seed"),
    "d.seed.hint": ("نفس البذرة تعطي نفس البيانات في كل مرة — لإمكانية التكرار",
                    "The same seed always reproduces the same dataset"),
    "d.engine": ("محرك التوليد", "Generation engine"),
    "d.engine.auto": ("تلقائي", "Automatic"),
    "d.engine.covariance": ("من مصفوفة التغاير", "From the covariance matrix"),
    "d.engine.scores": ("من درجات المتغيرات الكامنة", "From latent scores"),
    "d.engine.hint": (
        "التوليد من مصفوفة التغاير أدق في مطابقة التوزيعات؛ والتوليد من الدرجات "
        "ضروري عند وجود تفاعل بين متغيرين كامنين.",
        "Generating from the covariance matrix matches the distributions most "
        "precisely; latent scores are required when the model has an "
        "interaction term."),

    # ---- stage 4: power ----------------------------------------------- #
    "p.sample_size": ("حجم العينة", "Sample size"),
    "p.mode": ("طريقة التحديد", "How to decide"),
    "p.mode.suggest": ("اقترح الحجم الأمثل", "Suggest the optimal N"),
    "p.mode.fixed": ("حجم محدّد", "Fixed N"),
    "p.n": ("عدد الحالات (N)", "Number of cases (N)"),
    "p.target_power": ("القوة الإحصائية المستهدفة", "Target power"),
    "p.alpha": ("مستوى الدلالة α", "Significance level α"),
    "p.method": ("أسلوب التحليل", "Method"),
    "p.method.close_fit": ("اختبار التوفيق القريب (MacCallum وآخرون، 1996)",
                           "Test of close fit (MacCallum et al., 1996)"),
    "p.method.satorra": ("قوة كل مسار (Satorra-Saris)",
                         "Per-path power (Satorra-Saris)"),
    "p.method.monte_carlo": ("محاكاة مونت كارلو", "Monte Carlo"),
    "p.method.rule": ("القواعد الإرشادية", "Rules of thumb"),
    "p.rmsea_null": ("RMSEA الصفري", "Null RMSEA"),
    "p.rmsea_alt": ("RMSEA البديل", "Alternative RMSEA"),
    "p.mc_reps": ("عدد التكرارات", "Replications"),
    "p.df": ("درجات الحرية", "Degrees of freedom"),
    "p.free_params": ("عدد المعالم المقدَّرة", "Free parameters"),
    "p.moments": ("عدد العناصر في مصفوفة التغاير", "Observed moments"),
    "p.recommended": ("الحجم الموصى به", "Recommended N"),
    "p.achieved_power": ("القوة المتحققة", "Achieved power"),
    "p.curve": ("منحنى القوة", "Power curve"),
    "p.rule_per_param": ("حالات لكل معلمة", "Cases per parameter"),
    "p.rule_per_item": ("حالات لكل سؤال", "Cases per item"),
    "p.compute": ("احسب حجم العينة", "Compute sample size"),
    "p.explain": (
        "يستند الحساب إلى التعقيد الهيكلي للنموذج (درجات الحرية) لضمان قوة "
        "إحصائية كافية لاكتشاف التأثيرات المستهدفة، دون تضخيم العينة بغير داعٍ.",
        "The calculation uses the structural complexity of the model (its "
        "degrees of freedom) to reach adequate power for the effects you are "
        "targeting, without inflating N for no reason."),

    # ---- stage 5: targeting ------------------------------------------- #
    "t.fit_target": ("جودة التوفيق المستهدفة", "Target model fit"),
    "t.fit_target.explain": (
        "يستخدم البرنامج خوارزميات الهندسة العكسية لضبط مصفوفة التغاير بحيث "
        "تطابق النتائج مؤشرات التوفيق التي تختارها، مع الحفاظ على معالم النموذج "
        "كما حدّدتها بالضبط (طريقة Cudeck & Browne, 1992).",
        "The engine reverse-engineers the covariance matrix so the fitted "
        "indices land on the level you choose, while keeping your specified "
        "parameters as the exact minimiser (Cudeck & Browne, 1992)."),
    "t.enable_fit": ("تشغيل استهداف التوفيق", "Enable fit targeting"),
    "t.level": ("المستوى المستهدف", "Target level"),
    "t.method": ("أسلوب حقن الخطأ", "Model-error method"),
    "t.method.cudeck": ("Cudeck-Browne (يحافظ على المعالم)",
                        "Cudeck-Browne (parameter-preserving)"),
    "t.method.residual": ("اضطراب البواقي (أسرع)", "Residual perturbation (faster)"),
    "t.method.none": ("بلا حقن (توفيق مثالي)", "None (perfect fit)"),
    "t.tolerance": ("هامش التفاوت المقبول", "Tolerance"),
    "t.max_rounds": ("أقصى عدد لدورات المعايرة", "Maximum calibration rounds"),

    "fit.perfect": ("مثالي", "Perfect"),
    "fit.excellent": ("ممتاز", "Excellent"),
    "fit.good": ("جيد جدًا", "Good"),
    "fit.acceptable": ("مقبول", "Acceptable"),
    "fit.marginal": ("حدّي", "Marginal"),
    "fit.poor": ("ضعيف", "Poor"),
    "fit.custom": ("مخصّص", "Custom"),

    "t.significance": ("توجيه الدلالة الإحصائية", "Significance targeting"),
    "t.significance.explain": (
        "خصّص كل مسار ليكون دالًا إحصائيًا (p < 0.05) بقوة تأثير محددة، أو غير "
        "دال، بحسب السيناريو الذي ترغب في اختباره.",
        "Aim each path at statistical significance (p < 0.05) with a chosen "
        "effect size, or at non-significance, depending on the scenario you "
        "want to test."),
    "t.enable_significance": ("تشغيل توجيه الدلالة", "Enable significance targeting"),
    "t.null_coefficient": ("معامل المسارات غير الدالة",
                           "Coefficient for non-significant paths"),
    "t.max_resamples": ("أقصى عدد لإعادة السحب", "Maximum resamples"),
    "t.enforce_null": ("إلزام المسارات غير المستهدفة بعدم الدلالة",
                       "Require untargeted paths to stay non-significant"),
    "t.per_path": ("حالة كل مسار", "Per-path status"),
    "t.achieved": ("المتحقق", "Achieved"),
    "t.requested": ("المطلوب", "Requested"),
    "t.calibration": ("سجل المعايرة", "Calibration log"),
    "t.round": ("الدورة", "Round"),

    # ---- stage 6: results --------------------------------------------- #
    "r.results": ("النتائج", "Results"),
    "r.data_preview": ("معاينة البيانات", "Data preview"),
    "r.descriptives": ("الإحصاءات الوصفية", "Descriptive statistics"),
    "r.reliability": ("مؤشرات الثبات", "Reliability"),
    "r.validity": ("مؤشرات الصدق", "Validity"),
    "r.fit": ("مؤشرات جودة المطابقة", "Model fit indices"),
    "r.estimates": ("تقديرات المعالم", "Parameter estimates"),
    "r.loadings": ("التشبعات", "Loadings"),
    "r.paths": ("معاملات المسار", "Path coefficients"),
    "r.indirect": ("التأثيرات غير المباشرة", "Indirect effects"),
    "r.r_squared": ("معامل التحديد R²", "R-squared"),

    "r.alpha": ("ألفا كرونباخ", "Cronbach's alpha"),
    "r.cr": ("الثبات المركب", "Composite reliability"),
    "r.omega": ("أوميغا", "Omega"),
    "r.ave": ("متوسط التباين المستخلص", "Average variance extracted"),
    "r.sqrt_ave": ("الجذر التربيعي لـ AVE", "√AVE"),
    "r.convergent": ("الصدق التقاربي", "Convergent validity"),
    "r.discriminant": ("الصدق التمايزي", "Discriminant validity"),
    "r.fornell": ("معيار فورنيل-لاركر", "Fornell-Larcker criterion"),
    "r.htmt": ("نسبة التباين المشترك غير المتجانس HTMT",
               "Heterotrait-monotrait ratio (HTMT)"),
    "r.item_total": ("ارتباط السؤال بالدرجة الكلية", "Item-total correlation"),
    "r.alpha_dropped": ("ألفا عند حذف السؤال", "Alpha if item dropped"),
    "r.inter_item": ("متوسط الارتباط البيني", "Mean inter-item correlation"),

    "r.chi2": ("مربع كاي", "Chi-square"),
    "r.chi2_df": ("مربع كاي / درجات الحرية", "χ²/df"),
    "r.rmsea": ("جذر متوسط مربع خطأ التقريب RMSEA",
                "Root mean square error of approximation (RMSEA)"),
    "r.rmsea_ci": ("فترة ثقة RMSEA (90%)", "RMSEA 90% CI"),
    "r.pclose": ("احتمال التوفيق القريب", "p of close fit"),
    "r.cfi": ("مؤشر المطابقة المقارن CFI", "Comparative fit index (CFI)"),
    "r.tli": ("مؤشر توكر-لويس TLI", "Tucker-Lewis index (TLI)"),
    "r.nfi": ("مؤشر المطابقة المعياري NFI", "Normed fit index (NFI)"),
    "r.ifi": ("مؤشر المطابقة التزايدي IFI", "Incremental fit index (IFI)"),
    "r.srmr": ("جذر متوسط مربع البواقي المعياري SRMR",
               "Standardised root mean square residual (SRMR)"),
    "r.gfi": ("مؤشر جودة المطابقة GFI", "Goodness-of-fit index (GFI)"),
    "r.agfi": ("GFI المعدّل", "Adjusted GFI (AGFI)"),
    "r.aic": ("معيار أكايكي AIC", "Akaike information criterion (AIC)"),
    "r.bic": ("معيار بايز BIC", "Bayesian information criterion (BIC)"),

    "r.index": ("المؤشر", "Index"),
    "r.value": ("القيمة", "Value"),
    "r.threshold": ("المعيار", "Threshold"),
    "r.verdict": ("الحكم", "Verdict"),
    "r.interpretation": ("التفسير", "Interpretation"),
    "r.estimate": ("التقدير", "Estimate"),
    "r.se": ("الخطأ المعياري", "SE"),
    "r.z": ("القيمة الحرجة z", "z"),
    "r.p": ("مستوى الدلالة p", "p"),
    "r.std": ("المعياري", "Std."),
    "r.ci_lower": ("الحد الأدنى", "Lower"),
    "r.ci_upper": ("الحد الأعلى", "Upper"),
    "r.n": ("حجم العينة", "N"),
    "r.mean": ("المتوسط", "Mean"),
    "r.sd": ("الانحراف", "SD"),
    "r.median": ("الوسيط", "Median"),
    "r.min": ("الأدنى", "Min"),
    "r.max": ("الأعلى", "Max"),
    "r.missing": ("المفقود", "Missing"),
    "r.floor": ("نسبة أدنى فئة", "Floor %"),
    "r.ceiling": ("نسبة أعلى فئة", "Ceiling %"),
    "r.mardia": ("اختبار مارديا للاعتدالية المتعددة",
                 "Mardia's multivariate normality test"),
    "r.shapiro": ("اختبار شابيرو-ويلك", "Shapiro-Wilk test"),

    "verdict.pass": ("مستوفى", "Pass"),
    "verdict.fail": ("غير مستوفى", "Fail"),
    "verdict.borderline": ("حدّي", "Borderline"),
    "verdict.significant": ("دال", "Significant"),
    "verdict.not_significant": ("غير دال", "Not significant"),

    # ---- actions / status --------------------------------------------- #
    "act.ok": ("موافق", "OK"),
    "act.cancel": ("إلغاء", "Cancel"),
    "act.close": ("إغلاق", "Close"),
    "act.apply": ("تطبيق", "Apply"),
    "act.reset": ("استرجاع", "Reset"),
    "act.back": ("السابق", "Back"),
    "act.next": ("التالي", "Next"),
    "act.run": ("تشغيل المحاكاة", "Run simulation"),
    "act.rerun": ("إعادة التشغيل", "Run again"),
    "act.export": ("تصدير", "Export"),
    "act.copy": ("نسخ", "Copy"),
    "act.save_report": ("حفظ التقرير", "Save report"),
    "act.browse": ("استعراض…", "Browse…"),
    "act.yes": ("نعم", "Yes"),
    "act.no": ("لا", "No"),

    "status.ready": ("جاهز", "Ready"),
    "status.validating": ("جارٍ فحص النموذج…", "Validating…"),
    "status.compiling": ("جارٍ بناء مصفوفات النموذج…", "Building model matrices…"),
    "status.generating": ("جارٍ توليد البيانات…", "Generating data…"),
    "status.estimating": ("جارٍ تقدير النموذج…", "Estimating the model…"),
    "status.calibrating": ("جارٍ معايرة مؤشرات التوفيق…", "Calibrating fit…"),
    "status.reporting": ("جارٍ إعداد التقرير…", "Building the report…"),
    "status.done": ("تمّت المحاكاة بنجاح", "Simulation complete"),
    "status.cancelled": ("أُلغيت العملية", "Cancelled"),
    "status.failed": ("فشلت العملية", "Failed"),

    # ---- validation panel --------------------------------------------- #
    "v.title": ("فحص النموذج", "Model check"),
    "v.no_issues": ("لا توجد ملاحظات — النموذج جاهز للمحاكاة",
                    "No issues. The model is ready to simulate."),
    "v.errors": ("أخطاء", "Errors"),
    "v.warnings": ("تحذيرات", "Warnings"),
    "v.info": ("ملاحظات", "Notes"),
    "v.hint": ("الإجراء المقترح", "Suggested fix"),
    "v.where": ("الموضع", "Location"),
    "v.blocked": ("لا يمكن التشغيل قبل معالجة الأخطاء",
                  "Fix the errors before running"),
    "v.missing_ref": ("⚠ مرجع محذوف", "⚠ deleted reference"),

    # ---- dialogs ------------------------------------------------------ #
    "dlg.unsaved.title": ("تغييرات غير محفوظة", "Unsaved changes"),
    "dlg.unsaved.body": ("هل تريد حفظ التغييرات قبل المتابعة؟",
                         "Save your changes before continuing?"),
    "dlg.confirm_clear": ("سيُحذف النموذج الحالي بالكامل. هل تريد المتابعة؟",
                          "This clears the whole model. Continue?"),
    "dlg.export.title": ("تصدير البيانات والتقارير", "Export data and reports"),
    "dlg.export.formats": ("الصيغ", "Formats"),
    "dlg.export.include_latent": ("تضمين درجات المتغيرات الكامنة",
                                  "Include latent scores"),
    "dlg.export.include_shadow": ("تضمين القيم المتصلة قبل التفئية",
                                  "Include pre-categorisation values"),
    "dlg.export.lavaan": ("توليد صيغة lavaan لبرنامج R", "Emit lavaan syntax for R"),
    "dlg.export.mplus": ("توليد صيغة Mplus", "Emit Mplus syntax"),
    "dlg.export.directory": ("مجلد الحفظ", "Destination folder"),
    "dlg.export.done": ("تم التصدير إلى", "Exported to"),
    "dlg.about.body": (
        "بيئة محاكاة إحصائية متكاملة لتوليد بيانات تطابق الواقع الطبيعي، "
        "موجّهة لأبحاث العلوم الإنسانية والاجتماعية والسلوكية.",
        "An integrated statistical simulation environment for data that "
        "matches natural reality, aimed at research in the human, social and "
        "behavioural sciences."),

    # ---- example models ----------------------------------------------- #
    "ex.minimal": ("نموذج قياس أدنى (عاملان)", "Minimal two-factor CFA"),
    "ex.mediation": ("نموذج وساطة بسيط", "Simple mediation model"),
    "ex.second_order": ("نموذج هرمي من الدرجة الثانية", "Second-order model"),
    "ex.moderation": ("نموذج تعديل (تفاعل)", "Moderation model"),
    "ex.full": ("نموذج كامل معقّد", "Full complex model"),

    # ---- stage 2: complex realities (appended) ------------------------ #
    "x.auto_enable": ("تشغيل التوليد التلقائي للتحميلات المتقاطعة",
                      "Enable automatic cross-loading generation"),
    "x.magnitude_range.hint": (
        "يُسحب الحجم المطلق لكل تحميل متقاطع عشوائيًا من هذا المدى؛ "
        "والحد الأدنى لا يتجاوز الحد الأعلى أبدًا.",
        "The absolute magnitude of each generated cross-loading is drawn at "
        "random from this range; the low bound can never exceed the high one."),
    "x.self_loading": (
        "لا معنى لتحميل السؤال على عامله الأصلي (Home Factor)؛ اختر عاملًا آخر.",
        "A cross-loading onto the item's own home factor is meaningless; "
        "choose a different factor."),
    "x.same_item": (
        "لا يصح ربط خطأ السؤال بخطأ نفسه؛ اختر سؤالًا مختلفًا.",
        "An item cannot have a correlated error with itself; choose a "
        "different item."),
    "x.reason.hint": (
        "مبرّر نظري موجز يُذكر في التقرير، مثل تشابه صياغة السؤالين "
        "أو ورودهما في قسم واحد من الاستبانة.",
        "A short theoretical justification for the write-up: the two items are "
        "worded alike, or sit in the same block of the questionnaire."),
    "x.cross_loadings.empty": (
        "لا توجد تحميلات متقاطعة؛ نموذج القياس نظيف تمامًا",
        "No cross-loadings: the measurement model is perfectly clean"),
    "x.error_cov.empty": ("لا يوجد تغاير بين أخطاء القياس بعد",
                         "No correlated measurement errors yet"),
    "x.needs_items": (
        "أضف المتغيرات الكامنة وأسئلتها في المرحلة الأولى قبل إضافة التعقيدات",
        "Add constructs and their items in stage 1 before adding "
        "complications here"),
    "x.cmb_enable": ("تشغيل عامل طريقة القياس المشترك",
                     "Enable the common method factor"),
    "x.cmb_scope": ("نطاق عامل الطريقة", "Scope of the method factor"),
    "x.cmb_subset": ("أسئلة محددة فقط", "Selected items only"),
    "x.method_loadings": ("تشبعات عامل الطريقة", "Method-factor loadings"),
    "x.summary": ("حصيلة التعقيدات المضافة", "Complexity added to the model"),
    "x.df_cost": (
        "كل تعقيد تضيفه يُقدر كمعلمة إضافية، فيستهلك درجة حرية "
        "(Degree of Freedom) من ميزانية النموذج؛ ودرجات الحرية السالبة "
        "تعني نموذجًا غير مُعيَن.",
        "Every complication you add is one more estimated parameter, so it "
        "spends a degree of freedom from the model's budget; negative degrees "
        "of freedom mean an unidentified model."),

    # ---- stage 3 additions: distribution & generation engine ---------- #
    "d.clamp_feasible": (
        "تقييد الالتواء والتفلطح داخل المنطقة القابلة للتحقيق (Feasible Region)",
        "Clamp skewness and kurtosis to the feasible region"),
    "d.threshold_preview": ("معاينة العتبات ونسب الفئات",
                            "Threshold and category preview"),
    "d.apply_all_items": ("تطبيق على كل أسئلة النموذج", "Apply to every item"),
    "d.apply_to_construct": ("تطبيق على المتغير الكامن المحدَّد",
                             "Apply to the selected latent variable"),
    "d.thresholds.unavailable": (
        "محرّك العتبات (Thresholds Engine) غير متاح في هذه النسخة، لذا تُعرض "
        "نسب متساوية للفئات كتقدير مؤقت فقط.",
        "The thresholds engine is unavailable in this build, so equal category "
        "proportions are shown as a placeholder only."),
    "d.custom_style.hint": (
        "النمط المخصّص لا يحمل شكلًا بذاته؛ تُحدَّد نسب الفئات أو نقاط القطع "
        "لكل سؤال على حدة، وتُعرض هنا نسب متساوية.",
        "The custom style carries no shape of its own: set the proportions or "
        "thresholds item by item. Equal proportions are shown here."),
    "d.reproducibility": ("إمكانية التكرار", "Reproducibility"),
    "d.new_seed": ("بذرة جديدة", "New seed"),
    "d.engine.resolved": ("المحرّك الذي سيعمل فعليًا", "Engine that will run"),
    "d.engine.forced_scores": (
        "وجود تفاعل بين متغيّرين كامنين (Latent Interaction) يفرض التوليد من "
        "درجات المتغيرات الكامنة، لأن حاصل ضرب متغيّرين كامنين لا يمكن تمثيله "
        "في مصفوفة التغاير.",
        "A latent interaction forces generation from latent scores: the product "
        "of two latent variables cannot be represented in a covariance matrix."),
    "d.engine.auto_resolved": (
        "الاختيار التلقائي استقرّ على التوليد من مصفوفة التغاير، لأن النموذج "
        "لا يتضمّن تفاعلًا بين متغيّرين كامنين.",
        "The automatic choice settles on the covariance matrix, because the "
        "model has no latent interaction."),

    # ---- stage 5 additions: fit and significance targeting ------------ #
    "t.saturated": (
        "النموذج مشبع (Saturated Model): درجات الحرية صفر أو أقل، فلا توجد "
        "جودة توفيق يمكن استهدافها؛ أضف أسئلة أو قيّد بعض المعالم أولًا.",
        "The model is saturated: with zero or negative degrees of freedom "
        "there is no fit to target. Add items, or constrain parameters, "
        "first."),
    "t.level.implies": (
        "يستهدف هذا المستوى قيمة RMSEA = {rmsea} مع حد أدنى متوقع "
        "لمؤشر CFI = {cfi}.",
        "This level targets RMSEA = {rmsea} with an expected CFI floor "
        "of {cfi}."),
    "t.target_rmsea": ("قيمة RMSEA المستهدفة", "Target RMSEA"),
    "t.use_cfi": ("استهداف مؤشر CFI كذلك (هدف ثانوي)",
                  "Also target CFI (secondary goal)"),
    "t.target_cfi": ("قيمة CFI المستهدفة", "Target CFI"),
    "t.use_srmr": ("استهداف مؤشر SRMR كذلك (هدف ثانوي)",
                   "Also target SRMR (secondary goal)"),
    "t.target_srmr": ("قيمة SRMR المستهدفة", "Target SRMR"),
    "t.thresholds": ("المعايير المتعارف عليها لمؤشرات التوفيق",
                     "Conventional fit thresholds"),
    "t.no_paths": (
        "لا توجد مسارات أو تفاعلات بعد؛ أضفها في مرحلة بناء النموذج ثم عُد "
        "إلى هنا لتوجيه دلالتها.",
        "No paths or interactions yet: add them in the model-specification "
        "stage, then come back here to aim them."),
    "t.targets_summary": (
        "مستهدف أن يكون دالًا: {sig} — مستهدف أن يكون غير دال: {null} — "
        "من إجمالي {total} علاقة في النموذج.",
        "Targeted significant: {sig} — targeted non-significant: {null} — "
        "out of {total} relationships in the model."),
    "t.realised_rmsea": ("قيمة RMSEA المتحققة", "Realised RMSEA"),
    "t.rounds_used": ("دورات المعايرة المنفَّذة", "Calibration rounds used"),
    "t.within_tolerance": ("داخل هامش التفاوت المقبول", "Within tolerance"),
    "t.outside_tolerance": ("خارج هامش التفاوت المقبول", "Outside tolerance"),
    "t.paths_matched": ("تحقّقت كل المسارات المستهدفة",
                       "Every targeted path matched"),
    "t.paths_unmatched": ("لم تتحقّق كل المسارات المستهدفة",
                          "Not every targeted path matched"),
    "t.no_calibration": (
        "لم تُجرَ معايرة بعد؛ شغّل المحاكاة الكاملة ليظهر سجل المعايرة "
        "ونتائجها هنا.",
        "No calibration run yet: run the full simulation and the calibration "
        "log will appear here."),

    # ---- stage 4 additions: power analysis & sample size -------------- #
    "p.complexity": ("تعقيد النموذج", "Model complexity"),
    "p.settings": ("إعدادات تحليل القوة الإحصائية",
                   "Power analysis settings"),
    "p.results": ("نتائج تحليل القوة الإحصائية", "Power analysis results"),
    "p.df.negative": (
        "درجات حرية النموذج صفر أو سالبة، أي أن النموذج مشبَع (Saturated) أو "
        "غير مُعيَّن (Under-identified)، فلا توجد له مطابقة كلية يمكن اختبار "
        "قوتها. قلّل عدد المعالم المقدَّرة أو أضف أسئلة قبل حساب حجم العينة.",
        "The model has zero or negative degrees of freedom, so it is saturated "
        "or under-identified and has no global fit to have power against. "
        "Estimate fewer parameters, or add items, before sizing the sample."),
    "p.n.hint": (
        "يُستخدم هذا العدد كما هو عند اختيار «حجم محدّد»، أما عند اختيار "
        "«اقترح الحجم الأمثل» فيحسبه البرنامج بنفسه.",
        "This number is used as it stands under “Fixed N”; under “Suggest the "
        "optimal N” the engine works it out for you."),
    "p.rmsea.hint": (
        "يقارن اختبار التوفيق القريب بين فرض صفري عند RMSEA الصفري وفرض بديل "
        "عند RMSEA البديل، وكلما تقاربت القيمتان احتاج الاختبار عينة أكبر "
        "للتمييز بينهما.",
        "The test of close fit pits a null RMSEA against an alternative RMSEA; "
        "the closer the two values, the larger the sample the test needs to "
        "tell them apart."),
    "p.mc_reps.hint": (
        "كل تكرار يولّد عينة كاملة ويقدّر النموذج عليها، فالعدد الكبير يرفع "
        "دقة القوة المقدَّرة ويطيل زمن الحساب.",
        "Each replication generates a full sample and fits the model to it, so "
        "a high count buys precision at the cost of time."),
    "p.criterion": ("المحك", "Criterion"),
    "p.rules_of_thumb": ("القواعد الإرشادية لحجم العينة",
                         "Sample-size rules of thumb"),
    "p.kline_min": ("الحد الأدنى عند كلاين (200 حالة)",
                    "Kline's minimum (200 cases)"),
    "p.floor_5_to_1": ("الحد المطلق (5 حالات لكل معلمة)",
                       "Absolute floor (5 cases per parameter)"),
    "p.per_path": ("القوة الإحصائية لكل مسار بنائي",
                   "Power for each structural path"),
    "p.axis_power": ("القوة الإحصائية (Power)", "Statistical power"),
    "p.notes": ("ملاحظات على الحساب", "Notes on the calculation"),
    "p.apply_n": ("استخدم هذا الحجم في النموذج", "Apply this N to the model"),
    "p.computing": ("جارٍ حساب القوة الإحصائية وحجم العينة…",
                    "Computing power and sample size…"),
    "p.not_computed": (
        "اضغط «احسب حجم العينة» لتشغيل التحليل على النموذج بصيغته الحالية.",
        "Press “Compute sample size” to run the analysis on the model as it "
        "stands."),
    "p.unavailable": (
        "وحدة تحليل القوة الإحصائية غير متاحة في هذه النسخة، لذا تُعرض "
        "القواعد الإرشادية وحدها (10 حالات لكل معلمة مقدَّرة، و10 حالات لكل "
        "سؤال، والحد الأدنى عند كلاين وهو 200 حالة) دون منحنى القوة.",
        "The power analysis module is unavailable in this build, so only the "
        "rules of thumb are shown (10 cases per estimated parameter, 10 cases "
        "per item, and Kline's minimum of 200) with no power curve."),
    "p.failed": ("تعذّر إكمال تحليل القوة الإحصائية",
                 "The power analysis could not be completed"),
    "p.stale": (
        "تغيّر النموذج بعد إجراء هذا الحساب، والأرقام المعروضة تخص صيغته "
        "السابقة؛ أعد الحساب لتحديث الحجم الموصى به.",
        "The model has changed since this was computed, so the figures below "
        "describe its earlier form; run the calculation again to refresh the "
        "recommendation."),
}


# --------------------------------------------------------------------------- #
# Translator
# --------------------------------------------------------------------------- #


def _already_contains(primary: str, secondary: str) -> bool:
    """True when ``primary`` already carries the gist of ``secondary``.

    Compares the significant tokens -- words of three characters or more, minus
    bracketing -- so ``"...RMSEA"`` is recognised as already containing
    ``"Root mean square error of approximation (RMSEA)"`` only when every one of
    the latter's distinctive tokens is present.  An acronym counts on its own.
    """
    norm = primary.casefold()
    tokens = [w for w in re.split(r"[^\w\u0600-\u06FF]+", secondary) if len(w) >= 3]
    if not tokens:
        return False
    acronyms = [w for w in tokens if w.isupper() and w.isascii() and len(w) >= 2]
    if acronyms and all(a.casefold() in norm for a in acronyms):
        return True
    return all(w.casefold() in norm for w in tokens)


class Translator(QObject):
    """Holds the active language and notifies widgets when it changes."""

    language_changed = Signal(str)

    def __init__(self, language: str = ARABIC) -> None:
        super().__init__()
        self._language = language if language in LANGUAGES else ARABIC

    # -- state ------------------------------------------------------------ #

    @property
    def language(self) -> str:
        return self._language

    @property
    def is_rtl(self) -> bool:
        return self._language == ARABIC

    def set_language(self, language: str) -> None:
        if language not in LANGUAGES or language == self._language:
            return
        self._language = language
        self.language_changed.emit(language)

    def toggle(self) -> None:
        self.set_language(ENGLISH if self._language == ARABIC else ARABIC)

    # -- lookup ----------------------------------------------------------- #

    def __call__(self, key: str, **fmt: object) -> str:
        return self.t(key, **fmt)

    def t(self, key: str, **fmt: object) -> str:
        """Translate ``key`` into the active language.

        An unknown key returns ``"?key"`` rather than raising, so a typo shows
        up in the interface instead of taking a dialog down with it.
        """
        pair = STRINGS.get(key)
        if pair is None:
            return f"?{key}"
        text = pair[0] if self._language == ARABIC else pair[1]
        if not text:
            text = pair[1] or pair[0]
        if fmt:
            try:
                text = text.format(**fmt)
            except (KeyError, IndexError, ValueError):
                pass
        return text

    def both(self, key: str) -> str:
        """``"عربي (English)"`` -- used for headings where both terms help.

        Several Arabic entries already carry the English acronym inline
        (``"... RMSEA"``), and the English side often repeats it in brackets.
        Appending the other language there would read as a stutter, so the
        secondary term is dropped whenever its distinctive words are already
        present in the primary string.
        """
        pair = STRINGS.get(key)
        if pair is None:
            return f"?{key}"
        ar, en = pair
        if not ar or not en or ar == en:
            return ar or en
        primary, secondary = (ar, en) if self._language == ARABIC else (en, ar)
        if _already_contains(primary, secondary):
            return primary
        return f"{primary} ({secondary})"

    def enum(self, prefix: str, value: object) -> str:
        """Translate an enum member, e.g. ``enum("role", ConstructRole.MEDIATOR)``."""
        raw = getattr(value, "value", value)
        return self.t(f"{prefix}.{raw}")


#: Process-wide translator.  The UI uses this one; tests may build their own.
translator = Translator()


def tr(key: str, **fmt: object) -> str:
    """Module-level shortcut for ``translator.t``."""
    return translator.t(key, **fmt)


def set_language(language: str) -> None:
    translator.set_language(language)


def current_language() -> str:
    return translator.language


def is_rtl() -> bool:
    return translator.is_rtl


# --------------------------------------------------------------------------- #
# Catalogue hygiene (used by the test suite)
# --------------------------------------------------------------------------- #


def missing_translations() -> List[str]:
    """Keys whose Arabic or English side is empty."""
    return [k for k, (ar, en) in STRINGS.items() if not ar or not en]


def untranslated_keys() -> List[str]:
    """Keys whose two sides are byte-identical.

    A handful are legitimate -- a bare technical acronym reads the same in both
    languages -- so the test asserts a ceiling rather than zero.
    """
    return [k for k, (ar, en) in STRINGS.items() if ar == en]


def catalogue_size() -> int:
    return len(STRINGS)


__all__ = [
    "ARABIC", "ENGLISH", "LANGUAGES", "LANGUAGE_NAMES", "STRINGS",
    "Translator", "translator", "tr", "set_language", "current_language",
    "is_rtl", "missing_translations", "untranslated_keys", "catalogue_size",
]
