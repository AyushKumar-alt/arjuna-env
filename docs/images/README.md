# README screenshots

PNG assets are **not committed** in this repo: **Hugging Face Hub** rejects ordinary Git blobs for those files on Space pushes (use [Hub Xet](https://huggingface.co/docs/hub/xet) if you want binaries on HF Git).

The main **`README.md`** embeds images via **pinned `raw.githubusercontent.com` URLs** so they still render on **GitHub** and **HF Space**. To update screenshots:

1. Add new PNGs under `docs/images/` in a commit on **GitHub** `main`.
2. Note the commit SHA (e.g. `git rev-parse HEAD`).
3. Replace the SHA in the image URLs inside the root **`README.md`** (search for `raw.githubusercontent.com`).

Local copies of the originals may live elsewhere (e.g. under your `Pictures` folder).
