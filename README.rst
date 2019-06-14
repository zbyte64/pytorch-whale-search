
1) train vae on whale images (no artifacts/skew/rotation applied, 5 crop?)
- train on all
- 1/2 width squish (scale .5 along width always) (alternative is double wide autoencoder)
- add noise

2) learn rotation angle (transfer from vae)
- freeze vae
- inject rotation?
- set target / ideal whale fluke image
- style/content transfer loss
- model affines to minimize loss to target image ( affine_resample(vae(x)) = resampled images )
- resample from source with affine coords
- bonus: ignore water segments from target

3a) train new vae from affine samples only

3) train triplet loss
- identify(vae(afffine_resample(vae(x))))
- train on multiple instances only, augment data hard?
- annoy alternatives?

4) train classifier
- freeze triplet loss
- train on all
- cross entropy loss, weight classes
