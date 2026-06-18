# Data

FoliaNet trains on an ImageFolder-style dataset: a directory whose immediate
subfolders are class names, each containing images.

```
<root>/
    Corn_(maize)___Common_rust_/      *.jpg
    Corn_(maize)___Northern_Leaf_Blight/
    Corn_(maize)___healthy/
    Wheat___Leaf_rust/
    Wheat___healthy/
    ...
```

## Steps

1. Assemble or download a dataset in the layout above. Class-folder names should
   match the keys in `folianet/diseases.py` — add new entries there for any crops
   or diseases you introduce.
2. Set `data.root` in `configs/default.yaml` to the folder that directly contains
   the class subfolders.
3. Set `data.crops` to the crops you want to train on (a keyword filter on folder
   names), so you can keep a multi-crop dataset on disk and train on a subset.

If your data lives on Kaggle, you can fetch it with:

```bash
pip install kagglehub
python -m folianet.data.download --kaggle <owner>/<dataset-slug>
```

This directory is git-ignored — datasets are not committed.
