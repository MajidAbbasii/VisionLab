# Dataset provenance and licensing

## COCO / COCO 2017

VisionLab's experiments use COCO-derived image and annotation data. COCO's official terms distinguish annotations from the underlying images:

- COCO annotations and associated website content are offered under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Preserve attribution and comply with the license when redistributing those materials.
- The repository's `datasets/classes.txt` contains only the 80 COCO category names. The Windows package includes this small derived category-name list with attribution in [`THIRD-PARTY-NOTICES/COCO-ATTRIBUTION.md`](THIRD-PARTY-NOTICES/COCO-ATTRIBUTION.md); it does not include COCO image files or annotation JSON.
- COCO does not own the copyright to the Flickr-hosted images. Image use is subject to the applicable Flickr terms and the rights of the individual image copyright holders. The COCO terms place responsibility for obtaining permission to use/copy images on the user.
- COCO software/tools have separate software license terms; inspect the exact repository and files before copying any code.

Official sources (checked 2026-11-20):

- <https://cocodataset.org/dataset/termsofuse.htm>
- <https://creativecommons.org/licenses/by/4.0/>
- <https://github.com/cocodataset/cocoapi>

## VisionLab distribution policy

- Do not commit or redistribute COCO image files unless each image's rights and the distribution purpose have been checked.
- Do not assume that CC BY 4.0 for annotations grants rights to the underlying photographs.
- Keep dataset manifests, conversion scripts and YAML configuration separate from image/label payloads. A manifest or label file may still contain material whose redistribution terms need review.
- Record dataset name, version/split, download source/date, annotation format, conversion steps, selected image IDs and checksums for each experiment.
- For custom datasets, require the contributor to confirm they have rights to use and share the images and annotations under the intended terms.

This document summarizes upstream terms and is not legal advice. Re-check the official terms before each public dataset release.
