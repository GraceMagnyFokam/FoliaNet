"""
Helper to prepare a training dataset for FoliaNet.

FoliaNet trains on an ImageFolder-style dataset: a directory whose immediate
subfolders are class names (e.g. `Corn_(maize)___Common_rust_/`) containing images.

You can point it at any such dataset. If you keep your data on Kaggle, this helper
can pull it via `kagglehub` when you pass a dataset slug; otherwise it prints setup
instructions. After download, set `data.root` in configs/default.yaml to the folder
that directly contains the class subfolders.

    pip install kagglehub
    python -m folianet.data.download --kaggle <owner>/<dataset-slug>
"""

import argparse

INSTRUCTIONS = """
Dataset setup:
  1. Assemble (or download) an image dataset whose subfolders are class names, e.g.
       <root>/Corn_(maize)___Common_rust_/*.jpg
       <root>/Corn_(maize)___healthy/*.jpg
     Class-folder names should match the keys in folianet/diseases.py (add new
     entries there for any crops/diseases you introduce).
  2. Set data.root in configs/default.yaml to that <root> folder.
  3. Adjust data.crops in the config to the crops you want to train on.
"""


def main():
    ap = argparse.ArgumentParser(description="Prepare a FoliaNet training dataset.")
    ap.add_argument("--kaggle", help="Optional Kaggle dataset slug, e.g. owner/dataset-name")
    args = ap.parse_args()

    if not args.kaggle:
        print(INSTRUCTIONS)
        return

    try:
        import kagglehub
        path = kagglehub.dataset_download(args.kaggle)
        print(f"Downloaded to: {path}")
        print("Set data.root in configs/default.yaml to the subfolder containing class dirs.")
    except Exception as e:
        print(f"Download unavailable ({e}).")
        print(INSTRUCTIONS)


if __name__ == "__main__":
    main()
