# Licenses and Credits

This project incorporates code and models from multiple sources, each with their own licenses.

---

## Refocus ComfyUI Nodes

**License**: Apache License 2.0

The custom node code in this repository is licensed under Apache 2.0.

---

## Genfocus - Generative Refocusing

**Source**: https://github.com/rayray9999/Genfocus  
**Authors**: NYCU CP Lab (Chun-Wei Tuan Mu, Jia-Bin Huang, Yu-Lun Liu)  
**License**: Apache License 2.0  
**Paper**: [Generative Refocusing: Flexible Defocus Control from a Single Image](https://arxiv.org/abs/2512.16923)

### Genfocus LoRA Models

| Model | License | Source |
|-------|---------|--------|
| deblurNet.safetensors | Apache 2.0 | [HuggingFace](https://huggingface.co/nycu-cplab/Genfocus-Model) |
| bokehNet.safetensors | Apache 2.0 | [HuggingFace](https://huggingface.co/nycu-cplab/Genfocus-Model) |

```
Copyright 2025 NYCU CP Lab

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## FLUX.1-dev

**Source**: https://huggingface.co/black-forest-labs/FLUX.1-dev  
**Authors**: Black Forest Labs  
**License**: FLUX.1 [dev] Non-Commercial License  

### License Summary

- ✅ Research and personal use
- ✅ Non-commercial applications
- ❌ Commercial use without separate agreement
- ⚠️ Requires acceptance of license on HuggingFace

**Full License**: https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md

---

## Apple ml-depth-pro

**Source**: https://github.com/apple/ml-depth-pro  
**Authors**: Apple Inc.  
**License**: Apple Sample Code License  

### License Summary

- ✅ Research and development
- ✅ Personal use
- ⚠️ Limited redistribution rights
- ⚠️ Check full license for commercial use

**Full License**: https://github.com/apple/ml-depth-pro/blob/main/LICENSE

### Apple Sample Code License (Summary)

```
Copyright © 2024 Apple Inc.

This Apple software is provided for your personal, non-commercial use only. 
You may not redistribute the software without Apple's prior written consent.
```

---

## Third-Party Python Dependencies

| Package | License | Purpose |
|---------|---------|---------|
| diffusers | Apache 2.0 | FLUX pipeline |
| transformers | Apache 2.0 | Text encoding |
| peft | Apache 2.0 | LoRA adapter management |
| accelerate | Apache 2.0 | Model optimization |
| torch | BSD-3-Clause | Deep learning framework |
| safetensors | Apache 2.0 | Model file format |
| Pillow | HPND | Image processing |
| numpy | BSD-3-Clause | Numerical computing |

---

## Citation

If you use this project in research, please cite the Genfocus paper:

```bibtex
@article{Genfocus2025,
  title={Generative Refocusing: Flexible Defocus Control from a Single Image},
  author={Tuan Mu, Chun-Wei and Huang, Jia-Bin and Liu, Yu-Lun},
  journal={arXiv preprint arXiv:2512.16923},
  year={2025}
}
```

---

## Disclaimer

This project is provided "as is" without warranty of any kind. Users are responsible for:

1. Accepting all required model licenses (especially FLUX.1-dev)
2. Ensuring compliance with applicable licenses for their use case
3. Obtaining necessary permissions for commercial applications

The maintainers of this repository do not grant any rights to the underlying models beyond what is specified in their respective licenses.
