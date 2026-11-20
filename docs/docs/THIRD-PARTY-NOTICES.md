# Third-Party Notices

VisionLab is an aggregate of original project code and separately licensed third-party software, models and data. The GNU AGPL-3.0-only license in `LICENSE` applies to original VisionLab project source; it does not replace or override the license of any third-party component. `LICENSE` is the unmodified GNU/FSF text for version 3 (19 November 2007), retrieved from <https://www.gnu.org/licenses/agpl-3.0.txt> and byte-for-byte verified against that source on 20 November 2026.

## Direct Python dependencies

The project declares the following direct dependencies. Versions are constrained in `requirements.txt`; the exact environment snapshot is in `requirements-lock-py311-cu130.txt`. Confirm the license for the exact version and wheel/sdist being redistributed.

| Component | Declared package | License / notice |
|---|---|---|
| Ultralytics | `ultralytics` | AGPL-3.0; upstream also offers a separate Enterprise license. See <https://www.ultralytics.com/license>. |
| PyTorch | `torch` (installed separately for platform/CUDA) | BSD-style / multiple bundled third-party notices; inspect the exact wheel's `LICENSE` and `NOTICE` files. |
| Optuna | `optuna` | MIT; <https://github.com/optuna/optuna>. |
| OpenCV Python | `opencv-python` | Apache-2.0 for OpenCV 4.5.0 and later; wheel may include additional bundled components/notices. See <https://github.com/opencv/opencv-python>. |
| NumPy | `numpy` | BSD-3-Clause with separately licensed bundled components; see its distribution license files. |
| Pandas | `pandas` | BSD-3-Clause; inspect bundled dependency notices. |
| Matplotlib | `matplotlib` | Matplotlib license agreement (PSF/BSD-compatible components); retain its upstream license files. |
| PyYAML | `PyYAML` | MIT. |
| scikit-learn | `scikit-learn` | BSD-3-Clause; see <https://github.com/scikit-learn/scikit-learn>. |
| tabulate | `tabulate` | MIT. |
| pytest | `pytest` | MIT. |

## Direct .NET dependencies

| Component | NuGet package | License / notice |
|---|---|---|
| ONNX Runtime | `Microsoft.ML.OnnxRuntime` 1.20.1 | MIT; retain upstream license files from the NuGet package. |
| OpenCvSharp | `OpenCvSharp4` 4.10.0.20241108 | Apache-2.0. |
| OpenCvSharp Windows native runtime | `OpenCvSharp4.runtime.win` 4.10.0.20241108 | Apache-2.0 package metadata; separately review OpenCV native binaries and their bundled third-party notices. |

## Browser demo dependencies

| Component | Version / delivery | License / notice |
|---|---|---|
| ONNX Runtime Web | 1.23.2, loaded at runtime from jsDelivr (`ort.wasm.min.js` and WebAssembly assets) | MIT; source: <https://github.com/microsoft/onnxruntime>. The browser runtime is not vendored in this repository. |

The Apache License 2.0 text is included at [`THIRD-PARTY-NOTICES/Apache-2.0.txt`](THIRD-PARTY-NOTICES/Apache-2.0.txt), retrieved from the Apache Software Foundation's official text source.

The OpenCvSharp and OpenCV copyright attributions are recorded in [`THIRD-PARTY-NOTICES/OpenCvSharp-COPYRIGHT.txt`](THIRD-PARTY-NOTICES/OpenCvSharp-COPYRIGHT.txt); the upstream NuGet packages declare Apache-2.0. The native `OpenCvSharpExtern.dll` build configuration and its statically linked OpenCV third-party components are not fully identified by NuGet metadata; see [`THIRD-PARTY-NOTICES/OpenCvSharp-NATIVE-SOURCE.md`](THIRD-PARTY-NOTICES/OpenCvSharp-NATIVE-SOURCE.md). This is a separate unresolved native-binary review item.

The upstream OpenCV 4.10.0 Windows binary archive referenced by the OpenCvSharp Windows workflow recipe supplies a separate `etc/licenses` directory. Its 30 license/attribution files have been retained under [`THIRD-PARTY-NOTICES/OpenCV-4.10.0/`](THIRD-PARTY-NOTICES/OpenCV-4.10.0/), including FFmpeg, Intel IPP EULAs, ITT, libjpeg-turbo, OpenJPEG, libpng, libtiff, OpenEXR, protobuf and zlib. The matching Tesseract-vcpkg 2024.08.19 x64 archive supplied 18 `share/*/copyright` files, retained under [`THIRD-PARTY-NOTICES/Tesseract-5.3.1-vcpkg-2024.08.19/`](THIRD-PARTY-NOTICES/Tesseract-5.3.1-vcpkg-2024.08.19/). The native provenance record identifies the workflow recipe snapshot, archive tags, submodule commits, archive hashes and static-link inputs. The inspected OpenCvSharp commit is a README update, not a confirmed build commit, and no matching public Actions run was found; this evidence does not clear the Intel IPP EULA or prove the NuGet DLL was built by that workflow.

The FFmpeg Windows video backend is separately licensed under LGPL-2.1-or-later. Its copyright notice, license, OpenCV redistribution notes, and exact corresponding-source provenance are included in [`THIRD-PARTY-NOTICES/FFmpeg-COPYRIGHT.txt`](THIRD-PARTY-NOTICES/FFmpeg-COPYRIGHT.txt), [`THIRD-PARTY-NOTICES/FFmpeg-LGPL-2.1.txt`](THIRD-PARTY-NOTICES/FFmpeg-LGPL-2.1.txt), [`THIRD-PARTY-NOTICES/OpenCV-FFmpeg-README.txt`](THIRD-PARTY-NOTICES/OpenCV-FFmpeg-README.txt), and [`THIRD-PARTY-NOTICES/FFmpeg-SOURCE.md`](THIRD-PARTY-NOTICES/FFmpeg-SOURCE.md).

## Notable restored transitive packages requiring review

The current Windows/Python 3.11 and .NET restore inventory identifies these additional files and conditions:

- `Microsoft.ML.OnnxRuntime` 1.20.1: MIT package license plus a 345,046-byte `ThirdPartyNotices.txt` (6,508 lines). The exact native package contains `onnxruntime.dll` and `onnxruntime_providers_shared.dll` for Windows x64/x86/ARM64. NuGet Gallery metadata identifies release branch `rel-1.20.1`, source commit `5c1b7ccbff7e5141c1da7a9d963d660e5741c319` and build ID `610739`; the x64 native DLL embeds the matching short commit ID. Its notice includes Intel MKL's binary redistribution terms, zlib and protobuf BSD-style notices, the Eigen MPL-2.0 text, LLVM and other component notices. The binary also contains Eigen build/source markers. The complete Eigen 3.4.0 source candidate and exact MPL-2.0 text are staged under `THIRD-PARTY-SOURCES/ONNXRuntime-1.20.1/` and `Eigen-COPYING.MPL2.txt`, respectively. The independent Eigen pin is not Microsoft's build dependency lock. Intel MKL's applicability remains unconfirmed: no MKL DLL or identifying strings were found in the CPU package, but the sign-in-protected official build logs/configuration could not be inspected. These component/source questions remain a strict-audit blocker.
- `Microsoft.ML.OnnxRuntime.Managed` 1.20.1: MIT package license and the same upstream notice snapshot. The exact NuGet package contains managed .NET assemblies/PDBs but no `runtimes/*/native/*.dll`; its native component review is therefore tracked against `Microsoft.ML.OnnxRuntime`. The complete notice remains staged and included in the artifacts.
- `System.Memory` 4.5.5: MIT package license plus a 15,835-byte `THIRD-PARTY-NOTICES.TXT` covering Slicing-by-8/BSD, Unicode data, zlib, Mono/MIT, ISO, Intel/BSD, Xamarin/Novell/MIT, W3C, Brotli, Json.NET/MIT, LLVM and other attributions. The exact package files are retained under `THIRD-PARTY-NOTICES/NuGet/`.
- `System.Runtime.CompilerServices.Unsafe` 6.0.0: MIT package license plus a 46,990-byte `THIRD-PARTY-NOTICES.TXT` covering ASP.NET/Apache-2.0, Slicing-by-8/BSD, Unicode data, zlib and additional runtime notices. The exact package files are retained under `THIRD-PARTY-NOTICES/NuGet/`.
- `fonttools` 4.68.0: MIT for FontTools code. Its `LICENSE.external` documents OFL-1.1 test-font projects, but the installed distribution file inventory contains no font binaries; those test-font licenses therefore do not apply to font files redistributed by this project (none are bundled).
- `tqdm` 4.70.1: PyPI metadata declares `MPL-2.0 AND MIT`; its `LICENCE` maps the general files to MPL-2.0 and identifies the specific MIT exceptions. Retain that upstream file when distributing the dependency.
- `opencv_videoio_ffmpeg4100_64.dll` from `OpenCvSharp4.runtime.win` 4.10.0.20241108: OpenCV documents the Windows FFmpeg plugin as LGPL-2.1-or-later, distinct from OpenCV/OpenCvSharp's Apache-2.0 code. The exact DLL hash, OpenCV 4.10.0 binary pin, FFmpeg component versions, and corresponding source/build repository are recorded in `THIRD-PARTY-NOTICES/FFmpeg-SOURCE.md`.
- `OpenCvSharpExtern.dll` from the same package statically links a broad OpenCV module set, codec libraries, Intel IPP, ITT and Tesseract/Leptonica support libraries. The upstream linker inputs and source/workflow references are recorded in `OpenCvSharp-NATIVE-SOURCE.md`; the exact Tesseract-vcpkg copyright files are preserved, but IPP redistribution terms and package-to-CI binary attestation remain open before binary release.

The release packaging script copies the checked-in NuGet license and notice snapshots into the .NET package, so packaging does not depend on a mutable local NuGet cache. The staged snapshots were inventoried and SHA-256 hashed during this review; they preserve the staged upstream notice texts but do not themselves complete a component-level compatibility review of every bundled binary. In particular, the Eigen MPL-2.0 notice in ONNX Runtime must be evaluated against the actual component usage and applicable MPL source-availability terms; do not infer a project-wide AGPL relicensing requirement from the mere presence of the notice.

The FFmpeg corresponding-source archives for the OpenCV 4.10.0 Windows plugin are now retained under `THIRD-PARTY-SOURCES/FFmpeg-4.x_20240522/`, pinned to the upstream source-branch commit and accompanied by a SHA-256 manifest. The exact LGPL 2.1 text from the FFmpeg source archive is also preserved as `FFmpeg-COPYING.LGPLv2.1-upstream.txt` alongside the current GNU/FSF text. The release ZIP must include the complete source directory, both LGPL texts, the FFmpeg copyright notice, and OpenCV redistribution readme; source presence in the repository alone does not establish that a distributed binary recipient received it.

The Windows package contains only the 80 COCO category names in `datasets/classes.txt`, not images or annotation records. Their attribution and CC BY 4.0 reference are preserved in `COCO-ATTRIBUTION.md`.

Transitive Python and NuGet dependencies are version/platform dependent. Generate a fresh inventory with `python scripts/audit_licenses.py`; the report reads installed license files and verifies that known NuGet third-party notices have checked-in copies. The OpenCV 4.10.0 archive's own license directory and the NuGet package notices are checked in and included in package artifacts. The Tesseract-vcpkg copyright files are preserved, but their exact mapping to the shipped native DLL and the DLL's CI provenance remain unverified. Intel IPP redistribution, ONNX Runtime component obligations, and final LGPL source delivery/review remain explicit manual blockers. A clean metadata scan is not a legal opinion or a substitute for reviewing actual license texts and binary contents.

The Python wheel is a pure-Python `py3-none-any` wheel and does not vendor the 64 Python dependencies; those packages are installed separately and carry their own license/notice files in their own distributions. `results/reports/license-audit.json` records the exact installed Python/NuGet versions, license metadata, and upstream license-file paths. The Windows .NET ZIP does contain native NuGet/OpenCvSharp runtime binaries, so its packaging step copies the corresponding eight NuGet license/notice files, all 30 OpenCV archive notices, all 18 Tesseract-vcpkg copyright files, and the staged FFmpeg/Eigen corresponding-source packages.

## Model and dataset notices

- Ultralytics code and YOLO models: see `MODEL_CARD.md` and the [upstream licensing page](https://www.ultralytics.com/license). Ultralytics states that the full project must be under AGPL-3.0 or covered by Enterprise for use of its code/models/training pipeline; do not assume ONNX export changes model licensing.
- COCO annotations and images: see `DATASETS.md`. Annotation rights and image copyright are distinct.
- Do not add third-party weights, images, labels, datasets or native binaries to a release until their redistribution terms and required notices have been verified.

## Scope and limitations

This file is a project-level notice and dependency summary, not legal advice and not a complete binary SBOM. The exact licenses, copyright notices, attribution requirements and source-offer obligations must be checked for every release artifact and target platform.
