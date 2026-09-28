# Licensing and attribution inventory

Author-original software and its associated documentation are released under the [MIT License](../LICENSE), copyright 2026 Guodong Liang. This choice was approved by the author on 28 September 2026.

The MIT grant covers the author-original software and associated documentation, not a blanket license to dataset-derived research materials or third-party assets. Derived result tables and manifests are supplied to reproduce the reported analyses; the applicable source terms and attribution are listed below. No rights to underlying imagery, masks or model weights are granted by this software license. Third-party materials keep their own terms.

| Material | Package contents | Recorded source / terms |
|---|---|---|
| Author-original analysis/model code and associated documentation | Included | [MIT](../LICENSE), copyright 2026 Guodong Liang |
| Derived result tables and dataset manifests | Included for calculation reproduction | Separate from the software license; retain study and dataset attribution and observe applicable source terms below |
| Sen1Floods11 | Derived statistics and sample/event manifests; no images or masks | [Official repository](https://github.com/cloudtostreet/Sen1Floods11). Its README supplies dataset access/citation information. An explicit repository license was not located in this check; [issue 18](https://github.com/cloudtostreet/Sen1Floods11/issues/18) concerns the missing license. Do not apply another wrapper repository's license to these data. |
| HLS Burn Scars through GEO-Bench-2 | Derived results, sample manifest and normalization statistics | [Dataset card](https://huggingface.co/datasets/aialliance/burn_scars) declares CC BY 4.0 annotations, with original imagery terms; cites the [IBM–NASA source dataset](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars). |
| Kuro Siwo through GEO-Bench-2 | Derived results, sample/assets manifests and normalization statistics | [Dataset card](https://huggingface.co/datasets/aialliance/kuro_siwo) identifies CC BY annotations and Sentinel-1 open-access imagery; consult its linked original terms. |
| DOFA source | Not bundled; original license notice supplied for dependency attribution | [Official source](https://github.com/zhu-xlab/DOFA) under [MIT](https://github.com/zhu-xlab/DOFA/blob/master/LICENSE), copyright 2024 Zhitong Xiong; see `third_party/DOFA_LICENSE.txt`. |
| DOFA pretrained and study-trained weights | Not bundled | Download/redistribution terms and checkpoint access must be handled separately; the source-code license alone is not a blanket dataset license. |
| Python dependencies | Installed by the user | Their upstream licenses apply. The package uses Matplotlib's DejaVu Sans and does not ship proprietary font files. |

Dataset attribution should retain Bonafilia et al. (2020) for Sen1Floods11, Bountos et al. (2024) for Kuro Siwo, the IBM–NASA HLS Burn Scars dataset creators, and GEO-Bench-2 for the distributed external data format. Follow each linked provider's citation instructions. DOFA should be credited when using its architecture or weights.

Source pages checked on 28 September 2026. Accessibility or incomplete license metadata is not evidence that redistribution is unrestricted. The repository is [georisk-evaluation-audit](https://github.com/GuodongLiang1215/georisk-evaluation-audit).
