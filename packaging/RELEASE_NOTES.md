<div dir="rtl" align="right">

## محرك محاكاة بيانات النمذجة بالمعادلات البنائية — الإصدار 1.0.1

ملف تنفيذي واحد لويندوز 64-بت. **لا يحتاج تثبيت Python ولا أي تبعيات** — نزّله وشغّله.

### الجديد في 1.0.1

- إصلاح الخطأ الذي منع صفحات البرنامج من الفتح في الإصدار 1.0.0.
- كل الأزرار والأوامر تستجيب الآن: إما تنفذ عملها وتؤكده، أو تظهر معطّلة مع توضيح السبب.
- أسهم الترتيب تعمل على الأبعاد والأسئلة، و«تنويع التشبعات» يعطي قيمًا جديدة في كل ضغطة.
- إضافة المسارات والتفاعلات والتأثيرات غير المباشرة لا تكرر عنصرًا ولا تنشئ سلسلة غير صالحة.
- «تقدير النموذج» يعيد التقدير على البيانات الحالية بدل توليد بيانات جديدة.
- ضبط شكل الواجهة، واتجاه النص العربي داخل الجداول، وعرض الأخطاء غير المتوقعة في رسالة واضحة.

### التشغيل

نزّل `SEMSimulator.exe` واضغط عليه مرتين. الواجهة تفتح بالعربية افتراضيًا، ويمكن التبديل للإنجليزية من قائمة **عرض ← اللغة** دون إعادة تشغيل.

> **تنبيه مهم:** الملف **غير موقَّع رقميًا**، فسيعترض عليه Windows SmartScreen أول مرة برسالة
> «تم منع Windows تشغيل تطبيق غير معروف». اضغط **«مزيد من المعلومات» ← «تشغيل على أي حال»**.
> هذا متوقَّع لأي برنامج بلا شهادة توقيع، ولا علاقة له بسلامة الملف. للتأكد من أن ما نزّلته هو
> ما بُني فعلًا، قارن بصمة SHA-256 أدناه.

### ما الذي يفعله

| # | المرحلة |
|---|---------|
| ١ | **بناء مواصفات النموذج** — المتغيرات الكامنة وأبعادها وأسئلتها، والنماذج الهرمية من الدرجة الثانية، وتصنيف الأدوار وشبكة المسارات |
| ٢ | **محاكاة تعقيدات الواقع** — التحميلات المتقاطعة، والتغاير بين أخطاء القياس، وعامل طريقة القياس المشترك |
| ٣ | **توليد البيانات والتوزيعات** — نماذج العتبات لمقاييس ليكرت، والتحكم الكامل في الالتواء والتفلطح، وحقن التشويش |
| ٤ | **تحليل القوة وحجم العينة** |
| ٥ | **معايرة الدلالة وجودة التوفيق** |
| ٦ | **التقييم واستخراج المؤشرات** — ألفا كرونباخ والثبات المركب و AVE والصدق التمايزي و RMSEA و CFI و TLI و SRMR |

يصدّر البيانات بصيغ CSV و XLSX و SPSS ‏`.sav` مع تسميات المتغيرات والقيم، مع صيغة **lavaan** جاهزة للتشغيل في R وصيغة **Mplus**، وتقرير HTML مستقل بملف واحد.

</div>

---

## SEM Data Simulation Engine — v1.0.1

A single-file Windows 64-bit executable. **No Python installation, no dependencies** — download and run.

### What's new in 1.0.1

- Fixes the error that stopped the program's pages from opening in v1.0.0.
- Every button and menu command now responds: it does its work and confirms it, or is greyed out with the reason.
- The reorder arrows work on dimensions and items, and Vary loadings gives new values on each click.
- Adding paths, interactions and indirect effects no longer repeats an entry or builds an invalid chain.
- Estimate model now refits the current data instead of generating new data.
- Interface polish, correct direction for Arabic text in tables, and unexpected errors shown in a clear message.

The interface opens in Arabic by default; switch to English from **View → Language** without restarting.

> **The executable is not code-signed**, so Windows SmartScreen will warn on first launch
> ("Windows protected your PC"). Choose **More info → Run anyway**. That is expected for any
> unsigned application and says nothing about the file itself. To confirm you downloaded what
> was actually built, check the SHA-256 below.

### Verification

| | |
|---|---|
| File | `SEMSimulator.exe` |
| Size | about 106 MB |
| SHA-256 | `SHA256_PLACEHOLDER` |
| Built on | GitHub Actions, `windows-latest`, Python 3.12.10 |
| Build | [run RUN_ID_PLACEHOLDER](https://github.com/moali30/Data-Generation/actions/runs/RUN_ID_PLACEHOLDER) |
| Source | `COMMIT_PLACEHOLDER` |

Check it in PowerShell:

```powershell
Get-FileHash .\SEMSimulator.exe -Algorithm SHA256
```

The full test suite runs on the Windows runner before the build, and the finished bundle is then
verified by running a complete simulation inside it — compile, generate, estimate, fit indices,
reliability, power, report, and building every interface page:

```
python 3.12.10 on Windows (frozen bundle)
ok    compile: 14 items, df = 74
ok    validate: 0 error(s)
ok    generate: (400, 14)
ok    estimate + fit (exercises scipy): RMSEA = 0.0000, CFI = 1.0000, chi2 = 71.64 on 74 df
ok    reliability: Job satisfac alpha=0.819, Work engagem alpha=0.850, Task perform alpha=0.796
ok    power (exercises scipy.stats.ncx2): recommended N = 597
ok    report: 15049 bytes of HTML
ok    interface: 6 pages built
8/8 checks passed
```

Those numbers are identical to the same self-test run from source on Linux. Checking that a
packaged application merely *opens* proves very little: PyInstaller's static analysis misses the
scipy submodules that are imported lazily, so a broken bundle shows its menus quite happily and
fails the first time a p-value is computed — which is after the researcher has specified a model
and pressed Run. You can run that check yourself at any time:

```powershell
.\SEMSimulator.exe --self-test
```

### Command line

```powershell
.\SEMSimulator.exe                                      # Arabic, light theme
.\SEMSimulator.exe --lang en --theme dark               # English, dark theme
.\SEMSimulator.exe --example messy_reality              # open a bundled example
.\SEMSimulator.exe --list-examples
.\SEMSimulator.exe --self-test                          # verify the build
```

### Methods implemented

Fleishman (1978) and Vale & Maurelli (1983) for non-normal generation; threshold models for
ordinal responses; Cudeck & Browne (1992) for model error that keeps the specified parameters as
the exact minimiser; MacCallum, Browne & Sugawara (1996) and Satorra & Saris (1985) for power;
ML / ULS / DWLS estimation with polychoric correlations for ordinal data.

See [README.md](https://github.com/moali30/Data-Generation/blob/claude/charming-curie-ruokl8/README.md)
and [docs/ARCHITECTURE.md](https://github.com/moali30/Data-Generation/blob/claude/charming-curie-ruokl8/docs/ARCHITECTURE.md).
