# vendor/

## LivePortrait (not committed)
Clone of https://github.com/KlingAIResearch/LivePortrait at commit 9b294b3, unmodified.
Restore with:

    git clone https://github.com/KlingAIResearch/LivePortrait vendor/LivePortrait
    git -C vendor/LivePortrait checkout 9b294b3

Animal weights (v1.1, ~520 MB) go in
`vendor/LivePortrait/pretrained_weights/liveportrait_animals/base_models_v1.1/`:
appearance_feature_extractor.pth (3,386,487 B), motion_extractor.pth (112,542,562 B),
warping_module.pth (182,178,742 B), spade_generator.pth (221,811,030 B) from
https://huggingface.co/KwaiVGI/LivePortrait/tree/main/liveportrait_animals/base_models_v1.1

XPose / insightface weights are NOT needed: cropping uses our own tracks (track.py).

## lp_harness/ (committed)
smoke_lp.py is the verified pattern for running LivePortraitWrapperAnimal on Intel XPU
(torch 2.14.0+xpu, fp16, ~0.29 s/frame on Arc 140T). animate_cat.py follows it.
