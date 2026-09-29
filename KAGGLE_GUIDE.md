# Kaggle Submission Guide

This guide explains how to take your local work and submit it to the Kaggle competition.

## 1. Train and Save Model
Run your `train.ipynb` locally (or on a cloud GPU). This will produce a `saved_model/` folder containing:
- `pytorch_model.bin` (or `model.safetensors`)
- `config.json`
- `tokenizer.json` / `vocab.txt`
- etc.

## 2. Upload Model to Kaggle
1. Go to the [Kaggle Datasets](https://www.kaggle.com/datasets) page.
2. Click **New Dataset**.
3. Drag and drop your `saved_model/` folder (zipping it first usually helps: `zip -r model.zip saved_model/`).
4. Name it something like `quest-modernbert-model`.
5. Create the dataset.

## 3. Upload Source Code (Optional but Recommended)
Since you use `src/preprocess.py`, you need this code available in the notebook.
**Option A: Copy-Paste** -> Copy the contents of `preprocess.py` into a cell in `inference.ipynb`.
**Option B: Dataset** -> Upload the `src/` folder as a dataset (e.g., `quest-src-code`) and add it to your notebook inputs. Then `sys.path.append('/kaggle/input/quest-src-code')`.

## 4. Prepare Inference Notebook
1. Open your `inference.ipynb` on Kaggle (create a new notebook and copy-paste the code).
2. **Add Data**:
   - Add the competition dataset (`google-quest-challenge`).
   - Add your model dataset (`quest-modernbert-model`).
   - (Optional) Add your src dataset.
3. **Update Paths**:
   - Change `MODEL_PATH` to point to your uploaded model dataset (e.g., `/kaggle/input/quest-modernbert-model/saved_model`).
   - Change `pd.read_csv("test.csv")` to `/kaggle/input/google-quest-challenge/test.csv`.
4. **Turn Off Internet**: Code competitions usually require internet to be OFF for the submission run.
   - In the Kaggle Notebook sidebar (right side), find **Internet** and toggle it **OFF**.
   - Ensure your notebook installs/loads everything from local datasets (the inference notebook is designed for this).

5. **Select Accelerator**:
   - In the sidebar, set **Accelerator** to **GPU T4 x2** (or P100).
   - CPU inference will likely time out.

## 6. Submit
1. Click **Save Version** -> **Save & Run All (Commit)**.
2. Once finished, go to the **Viewer** for that version.
3. Scroll down to `submission.csv` in the output section.
4. Click **Submit**.

