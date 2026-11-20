# VisionLab — آزمایشگاه تشخیص اشیا با YOLO11

**راهنمای کامل دموی وب:** [docs/README.fa.md](docs/README.fa.md) · **راهنمای انگلیسی:** [README.md](README.md) · **دموی آنلاین:** [majidabbasii.github.io/VisionLab](https://majidabbasii.github.io/VisionLab/)

## معرفی

VisionLab یک پروژه عملی و تحقیقاتی در حوزه Computer Vision و Object Detection است. هدف آن نمایش چرخه کامل توسعه یک مدل هوش مصنوعی است: آماده‌سازی داده، آموزش، ارزیابی، آزمایش و بهینه‌سازی، خروجی ONNX و اجرای مدل در Python، C#/.NET و مرورگر.

مدل پایه پروژه **YOLO11n** است؛ نسخه سبک YOLO11 برای تشخیص اشیا. مدل فعلی ۸۰ دسته COCO را تشخیص می‌دهد و ورودی استاندارد آن `640×640` است.

## قابلیت‌های اصلی

### دموی آنلاین در مرورگر

- تشخیص اشیا در تصاویر JPG، PNG و WEBP
- پردازش فایل‌های ویدیویی MP4، WEBM و OGG
- تشخیص زنده اشیا با وب‌کم و مجوز دوربین مرورگر
- نمایش Bounding Box، نام کلاس و Confidence
- تنظیم Confidence Threshold و NMS IoU Threshold
- تحلیل نتایج: تعداد اشیا، تعداد کلاس‌ها، میانگین Confidence، زمان Inference، نرخ پردازش و توزیع کلاس‌ها
- پردازش محلی در مرورگر بدون ارسال تصاویر یا فریم‌ها به سرور پروژه

برای جزئیات محدودیت‌ها، روش استفاده، نیازمندی‌های مرورگر و حریم خصوصی، [راهنمای فارسی دموی وب](docs/README.fa.md) را ببینید.

### برنامه‌های Python و C#/.NET 8

- تشخیص تصویر و پردازش دسته‌ای پوشه‌ها
- پردازش فایل ویدیو و دریافت فریم از وب‌کم
- اجرای مدل ONNX در برنامه C# با ONNX Runtime و OpenCvSharp
- ذخیره تصاویر حاشیه‌نویسی‌شده و گزارش‌های JSON

### پژوهش و آموزش مدل

- آموزش و Fine-tuning مدل YOLO11 با Python، Ultralytics و PyTorch
- ارزیابی با Precision، Recall، mAP50 و mAP50-95
- بهینه‌سازی اکتشافی Hyperparameter با Optuna
- Sensitivity Analysis برای Confidence، IoU و Image Size
- Ablation Study برای بررسی اثر تنظیمات Data Augmentation
- تبدیل Checkpoint به ONNX و مقایسه نتایج Runtimeها

## معماری

```text
COCO-derived dataset
        │
        ▼
Python · Ultralytics / PyTorch
Training · Evaluation · Optuna · Ablation
        │
        ▼
      ONNX model
        │
        ├── Python + ONNX Runtime
        ├── C# / .NET 8 + ONNX Runtime + OpenCvSharp
        └── Browser + ONNX Runtime Web + WebAssembly
```

## داده‌ها و نتایج ثبت‌شده

تقسیم‌بندی آزمایش داده توسعه‌یافته شامل ۲۰٬۰۰۰ تصویر آموزش، ۵٬۰۰۰ تصویر اعتبارسنجی و ۵٬۰۰۰ تصویر آزمون است. برنامه آموزش ۱۰۰ دوره‌ای به درخواست کاربر در دوره ۱۰ متوقف شد و بهترین Checkpoint انتخاب‌شده از دوره اول است؛ بنابراین آموزش کامل ۱۰۰ دوره‌ای انجام نشده است.

| معیار آزمون | نتیجه |
|---|---:|
| Precision | 0.48972 |
| Recall | 0.35184 |
| mAP50 | 0.34698 |
| mAP50-95 | 0.22623 |

در تست تطبیق Python و C# روی ۱۰ تصویر، ۳۱ تشخیص از ۳۱ تشخیص هر دو Runtime تطبیق داده شدند و میانگین IoU جعبه‌ها 0.97591 بود. این نتیجه فقط نمونه آزمایش‌شده را توصیف می‌کند و تضمین عمومی نیست.

## تست‌های ثبت‌شده

- ۱۸ تست Python موفق
- ۹ تست Integration در .NET موفق
- Build نسخه Release برنامه .NET با صفر Warning و صفر Error در گزارش ثبت‌شده
- تست هفت شکل تصویر در C# با هفت خروجی JSON معتبر
- تست پردازش ویدیوی C# با ۶۰ فریم در آخرین بررسی

## اجرای محلی Demo وب

از ریشه مخزن:

```powershell
py -m http.server 8080 --directory docs
```

سپس `http://localhost:8080` را باز کنید. برای وب‌کم، مرورگر باید روی HTTPS یا localhost باشد و مجوز دوربین داده شود.

## محدودیت‌های مهم

- مدل فعلی فقط ۸۰ کلاس COCO را هدف می‌گیرد و Open-Vocabulary نیست.
- Tracking، Segmentation و Pose Estimation در نسخه فعلی پیاده‌سازی نشده‌اند.
- Demo وب از WebAssembly CPU استفاده می‌کند؛ WebGPU فعال نشده است.
- سرعت ویدیو و وب‌کم به سخت‌افزار و مرورگر بستگی دارد و ممکن است همه فریم‌های منبع پردازش نشوند.
- خروجی ویدیوی WEBM ضبط‌شده از Canvas، صدای ویدیوی ورودی را شامل نمی‌شود.
- مدل برای تصمیم‌گیری پزشکی، ایمنی‌حیاتی یا سایر کاربردهای حساس اعتبارسنجی نشده است.
- نتایج Optuna، Sensitivity و Ablation اکتشافی هستند و برای ادعاهای علمی قوی‌تر به تکرارهای بیشتر نیاز دارند.

## ساختار مهم مخزن

```text
configs/              تنظیمات مدل، آموزش و آزمایش‌ها
datasets/             تنظیمات و داده‌های مجاز/مستندشده
models/               Checkpointها و خروجی ONNX
experiments/           اجراهای آموزش و دیتابیس Optuna
results/               معیارها، نمودارها، پیش‌بینی‌ها و گزارش‌ها
src/visionlab/         موتور پژوهشی Python
src/VisionLab.Core/    هسته تشخیص ONNX در .NET
src/VisionLab.App/     برنامه خط فرمان C#
docs/                 سایت GitHub Pages و دموی مرورگر
tests/                تست‌های Python و .NET
```

## نسخه و مجوز

نسخه مستندشده فعلی `v1.0.0-rc1` (Release Candidate) است، نه نسخه پایدار نهایی. کد اصلی پروژه تحت GNU AGPL-3.0-only است. وزن‌های مدل، اجزای Ultralytics، وابستگی‌ها و داده‌های COCO ممکن است شرایط جداگانه داشته باشند. پیش از بازتوزیع مدل یا بسته‌های اجرایی، فایل‌های `LICENSE`، `THIRD-PARTY-NOTICES.md`، `DATASETS.md` و `MODEL_CARD.md` و الزامات Corresponding Source را بررسی کنید. این توضیحات مشاوره حقوقی نیستند.
