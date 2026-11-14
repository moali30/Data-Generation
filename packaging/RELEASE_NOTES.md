<div dir="rtl" align="right">

## محرك محاكاة بيانات النمذجة بالمعادلات البنائية — الإصدار 1.5.0

ملف تنفيذي واحد لويندوز 64-بت. **لا يحتاج تثبيت Python ولا أي تبعيات** — نزّله وشغّله.

### الجديد في 1.5.0

- الارتباط بين المتغيرات الديموغرافية: كارت جديد في تبويب الديموغرافيا تربط فيه متغيرين بيمشوا مع بعض في العينات الحقيقية (السن وسنين الخبرة، المؤهل والمستوى الوظيفي). بتختار قوة الارتباط (ضعيف، متوسط، قوي، أو قيمة V لكرامر بالظبط)، والفئات اللي بتجتمع مع بعض، وهل اختبار كاي تربيع مطلوب يطلع دال. قبل كده كل متغير كان مستقل تمامًا عن التاني، فكاي تربيع بينهم كان بيطلع غير دال دايمًا.
- كل متغير بيحافظ على نِسب فئاته، والمتغيرات اللي مش مربوطة والإجابات نفسها مش بتتغير. والمتغيرات المربوطة بطريق غير مباشر بترتبط بالقدر اللي بتفرضه الروابط (السن مع الخبرة والخبرة مع الوظيفة، فالسن بيرتبط بالوظيفة شوية).
- لو المتغيرين المربوطين ضابطين، النموذج بقى يعرف الارتباط بينهم فعلًا، فالمعايرة بتوصل لهدف المطابقة. ولو متغير المقارنة ليه فروق ومربوط بمتغير تاني، الارتباط بيفضل موجود بعد توزيع الفروق.
- النتائج والتقرير فيهم جدول جديد لاختبار كاي تربيع ومعامل V لكل ارتباط، وقالب الذكاء الاصطناعي فيه مفتاح جديد `associations`.

### الجديد في 1.4.0

- الفروق بين الفئات الديموغرافية: المتغير اللي دوره «اختبار فروق» بقى ليه قسم «الفروق بين الفئات»، تختار فيه المتغير الكامن، وحجم الفرق (صغير، متوسط، كبير، أو قيمة d / مربع إيتا بالظبط)، والفئة الأعلى، وهل الفرق مطلوب يطلع دال ولا لأ. قبل كده الفئات كانت بتتوزع بمعزل عن الإجابات، فاختبار t وتحليل التباين كانوا بيطلعوا غير دالين دايمًا، وده مستحيل في بيانات حقيقية.
- الإجابات نفسها ونتائج النموذج مش بتتغير: البرنامج بيوزّع المستجيبين على الفئات حسب إجاباتهم، مع الاحتفاظ بعدد كل فئة. والمتغيرات اللي مالهاش فرق محدد بتختلف بقدر ارتباطها بالمتغير اللي له فرق، زي البيانات الحقيقية. وتقدر تثبّت متغير مرتبط على «لا يوجد فرق».
- قالب الذكاء الاصطناعي فيه مفتاح جديد `differences` لمتغيرات المقارنة، والمساعد بيتوجّه يضيفه لما يكون عندك فرض عن فروق ديموغرافية.

### الجديد في 1.3.1

- إصلاح مهم في تقدير النموذج: في بعض النماذج (مثلًا أربع متغيرات فيها متغيرين مستقلين مترابطين) خوارزمية التقدير كانت بتقف عند نقطة البداية وتعتبر نفسها نجحت، فالمسارات كانت بتطلع **نفس الأرقام اللي أدخلتها بالظبط** (0.30 و0.50...) بدل ما تتقدّر من البيانات، ومؤشرات المطابقة كانت بتتحسب عند القيم دي فتطلع سيئة (RMSEA 0.066 و SRMR 0.23 بدل الهدف). دلوقتي التقدير بيتحرك فعلًا من البيانات، والمعايرة بتوصل للهدف.
- الأسئلة معكوسة الصياغة بقت واقعية: العينة الراضية بقت **ترفض** عبارة زي «لست سعيدًا بالخدمة» (متوسطها الخام حوالي 2)، فبعد إعادة الترميز بتتماشى مع باقي أسئلة المتغير. وتشبعها بقى أضعف شوية من الأسئلة الإيجابية زي ما بيحصل في البيانات الحقيقية.
- «غير موافق بشدة» بقت بتظهر في كل سؤال: إعداد جديد «الإجابات عند أطراف المقياس» بيوزّع نسبة صغيرة من الإجابات على كل الفئات (5% في المستوى الواقعي)، بدل ما يبقى في أسئلة محدش اختار فيها 1 أبدًا.

### الجديد في 1.3.0

- البيانات بقت شبه البيانات اللي بتتجمع فعلًا. قبل كده كل أسئلة المتغير كان ليها نفس المتوسط (3.00 مثلًا) ونفس الانحراف المعياري ونفس التشبع، والتوزيع متماثل تمامًا، وده بيكشف إن البيانات مولَّدة. دلوقتي البرنامج بيسحب «مجتمع واقعي» حوالين نموذجك: كل متغير ليه مستوى خاص على المقياس، وكل سؤال ليه صعوبة وتشتت خاصين، والتشبعات والمسارات بتتفاوت شوية حوالين القيم اللي حددتها من غير ما إشارتها أو الدلالة المطلوبة تتغير.
- إعداد جديد في صفحة «توليد البيانات والتوزيعات» اسمه «التباين الواقعي بين الأسئلة والمتغيرات»: بدون، خفيف، واقعي (الافتراضي)، قوي، أو مخصّص بخمس قيم.
- العينة الديموغرافية بقت تتسحب عشوائيًا بالنسب المطلوبة افتراضيًا، فالأعداد بتطلع زي العينات الحقيقية (317 و204 و158 بدل 320 و200 و160). الحصص الدقيقة لسه موجودة لو العينة حصصية فعلًا.
- قالب الذكاء الاصطناعي فيه مفتاح جديد `realism.variation`، والحصص العشوائية بقت الافتراضية، والمساعد بيتوجّه يختار نمط الاستجابة المناسب لكل متغير (مقاييس الرضا والثقة والنية بتميل للموافقة) بدل ما يخلي كل المتغيرات متماثلة حوالين المنتصف.

### الجديد في 1.2.3

- إصلاح قفل البرنامج فجأة أول ما تبدأ المحاكاة (في 1.2.1 و 1.2.2). شريط التقدم كان بيتحدّث من خيط الحساب بدل الواجهة، فالبرنامج كان بيقفل من غير أي رسالة.

### الجديد في 1.2.2

- النماذج اللي فيها أثر تعديل (تفاعلات) بقت توصل لهدف المطابقة زي أي نموذج تاني. قبل كده كانت البيانات بتطابق النموذج تقريبًا تمامًا، والـ RMSEA بيطلع قريب من الصفر مهما كان الهدف، والمعايرة مكانتش بتقدر تحركه. أثر التعديل نفسه فضل زي ما هو.

### الجديد في 1.2.1

- زر «إيقاف» يوقف المحاكاة فعلًا: الأزرار ترجع فورًا، والحساب يتوقف خلال جزء من الثانية. قبل كده كان الإيقاف يُتجاهل أثناء المعايرة والمحاكاة تكمل.
- شريط الحالة يعرض الوقت المنقضي والوقت المتبقي المتوقع لانتهاء المحاكاة، ويتحدّث كل ثانية. شريط التقدم بقى يتحرك أثناء حساب القوة الإحصائية وكل سحبة في المعايرة بدل ما يقف.
- مخطط المسار اتعاد رسمه: المتغيرات المعدِّلة في صف فوق أو تحت النموذج وسهمها ينزل على المسار اللي بتعدّله، والوسيط في مكانه بين المتغيرات اللي بتأثر فيه وبين التابع، والأسماء الطويلة جوه الشكل، ومعاها الرمز والدور، ومفتاح للرسم.
- قالب الذكاء الاصطناعي بقى يشرح للمساعد إزاي يكتب أثر التعديل والارتباطات بالشكل الصحيح، عشان الرسمة تطلع مظبوطة. ولو المساعد كتب التعديل غلط (مسار مباشر بدل تعديل، أو معدِّل مش بيعدّل حاجة) البرنامج بيرجّعه بالخطأ عشان يصلحه.

### الجديد في 1.2.0

- قالب JSON للذكاء الاصطناعي من قائمة ملف: احفظه أو انسخه، وأرسله إلى أي مساعد مع شرح البيانات التي تحتاجها، ثم استورد ما يرجعه من ملف أو من الحافظة.
- لو في القالب أخطاء يعرضها البرنامج كلها بمكانها، وزر واحد ينسخها لترسلها إلى المساعد ليصححها.

### الجديد في 1.1.0

- تبويب جديد للديموغرافيا والمتغيرات الضابطة: تحدد المتغيرات وفئاتها بحرية، ودور كل متغير (متغير ضابط، وصف إحصائي، اختبار فروق، تحليل متعدد المجموعات).
- نسب دقيقة لكل فئة بأسلوب العينة الحصصية، أو سحب عشوائي مرجّح.
- المتغير الضابط يدخل النموذج الهيكلي بمسارات ترسمها أنت، وتتحكم في حجم أثرها ومعنويتها دون أن تتغير العلاقات الأساسية.
- صفحة النتائج تعرض التكرارات المطلوبة والمحققة، واختبارات T و ANOVA، وثبات النموذج عبر المجموعات.
- تصدير البيانات والتقرير وأكواد lavaan و Mplus يشمل المتغيرات الديموغرافية.

### الجديد في 1.0.2

- تغيير رتبة المتغير إلى «الدرجة الثانية» كان يمسح أسئلته ويترك النموذج من الدرجة الأولى. الآن تُوزَّع الأسئلة على ثلاثة أبعاد وتظهر داخلها.
- اختيارات القوائم، مثل دور المتغير ونوع المقياس، تُسجَّل الآن بقيمتها الصحيحة.
- لوحات التعديل لم تعد تُقص أو تحتاج تمريرًا أفقيًا عند أصغر حجم للنافذة.

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

## SEM Data Simulation Engine — v1.5.0

A single-file Windows 64-bit executable. **No Python installation, no dependencies** — download and run.

### What's new in 1.5.0

- Associations between demographic variables: a new card on the demographics tab links two variables that go together in real samples (age and years of experience, education and job level). You choose the strength (weak, moderate, strong, or an exact Cramér's V), which categories go together, and whether the chi-square test should be significant. Until now every variable was independent of every other, so every chi-square between them came out non-significant.
- Every variable keeps its category shares, and the unlinked variables and the answers do not change. Variables linked only through others go together as far as the links imply (age with experience and experience with job level makes age go a little with job level).
- When both linked variables are controls, the model now knows how they correlate, so calibration still reaches its fit target. A comparison variable with group differences keeps its association after it is regrouped.
- Results and the report gain a chi-square and Cramér's V table for each link, and the AI template a new `associations` key.

### What's new in 1.4.0

- Group differences on demographics: a variable with the Group comparison role now has a "Group differences" section where you choose the construct, the size of the difference (small, medium, large, or an exact d or eta squared), which group scores highest, and whether the test should come out significant. Until now groups were assigned apart from the answers, so every t-test and ANOVA came out non-significant, which no real sample does.
- The answers and the model results do not change: respondents are placed in groups according to their answers, and each group keeps its size. Constructs with no difference of their own differ as far as they correlate with one that has, as in real data, and a correlated construct can be held level with "None".
- The AI template has a new `differences` key for comparison variables, and the assistant is told to add it when you hypothesise demographic differences.

### What's new in 1.3.1

- An important estimation fix: in some models (four constructs with two correlated independents, for example) the optimiser stopped at its starting point and reported success, so the paths came back as **exactly the numbers you entered** (0.30, 0.50...) instead of being estimated from the data, and the fit indices, computed at those values, came out poor (RMSEA 0.066 and SRMR 0.23 instead of the target). The estimates now really come from the data, and calibration reaches its target.
- Reverse-worded items behave realistically: a satisfied sample now mostly *disagrees* with "I am unhappy with the service" (raw mean about 2), so after recoding it lines up with the rest of its construct, and it loads a little more weakly than the positively worded items, as real reversed items do.
- "Strongly disagree" now turns up on every item: a new "Answers at the scale ends" amount spreads a small share of answers over every category (5% at the realistic level), instead of leaving items nobody ever answered 1 on.

### What's new in 1.3.0

- Generated data now look collected rather than simulated. Every item of a construct used to share the same mean (3.00, say), the same standard deviation and the same loading, in a perfectly symmetric distribution, which gave the data away. The program now draws one realistic population around your model: each construct has its own level on the scale, each item its own difficulty and spread, and the loadings and paths scatter a little around the values you set without changing sign or the significance you asked for.
- A new "Real-world variation" setting on the Distributions page: none, mild, realistic (the default), strong, or custom with five amounts.
- Demographic categories are now drawn at random with the requested shares by default, so the counts vary the way a real sample's do (317, 204, 158 rather than 320, 200, 160). Exact quotas remain available for a genuine quota sample.
- The AI template has a new `realism.variation` key, defaults to random quotas, and steers the assistant to choose each construct's response style on its own merits (satisfaction, trust and intention scales lean towards agreement) instead of centring every construct on the midpoint.

### What's new in 1.2.3

- Fixes the program closing without a message as soon as a simulation started (in 1.2.1 and 1.2.2). The progress bar was being updated from the computation thread instead of the interface.

### What's new in 1.2.2

- Models with moderation (interactions) now reach their fit target like any other model. Their data used to fit almost perfectly, with RMSEA near zero whatever the target, and calibration could not move it. The moderation effects themselves are unchanged.

### What's new in 1.2.1

- Stop really stops a run: the controls come back at once and the computation halts within a fraction of a second. It used to be ignored during calibration, and the run carried on.
- The status bar shows the time elapsed and the estimated time left, updated every second. The progress bar now moves during the power analysis and with every calibration draw instead of freezing.
- The path diagram is redrawn: moderators sit in a band above or below the model with their arrow landing on the path they moderate, a mediator sits between what feeds it and the outcome, long names fit inside their ellipse with their code and role, and a legend explains the lines.
- The AI template now tells the assistant exactly how to write a moderation and a correlation, so the diagram comes out right. A moderation written wrongly (a plain path instead of an interaction, or a moderator that moderates nothing) is sent back with the problem to fix.

### What's new in 1.2.0

- An AI template in the File menu: save or copy it, send it to any assistant with a description of the data you need, and import the answer from a file or the clipboard.
- Problems in a filled template are all listed with their locations, and one button copies them to send back to the assistant.

### What's new in 1.1.0

- A new Demographics and controls tab: define variables and their categories freely, and give each one a role (control, descriptive, group comparison, multi-group analysis).
- Exact category percentages as a quota sample, or weighted random sampling.
- Control variables enter the structural model through paths you draw, with their effect size and significance under your control and the substantive relationships left as specified.
- The results page shows requested and achieved frequencies, T-tests and ANOVA, and multi-group invariance.
- Data exports, the report and the lavaan and Mplus syntax now include the demographics.

### What's new in 1.0.2

- Switching a construct to second order used to delete its items and leave the model first-order. The items are now dealt into three dimensions and shown inside them.
- Choices in the drop-down lists, such as a construct's role or a scale type, are now stored as their proper values.
- The editing panels are no longer cut off or scrolled sideways at the smallest window size.

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
