### Steps:
0. Create a `data/` folder in the main directory
For each 'category'
1. Create a subfolder of `data/` with the dataset name
2. Trim the audio to at most 15 seconds with `trim_audio.sh`
3. Get the embeddings using either `run.py` or `embeddings/get_embeddings.py`
4. Analyze them with `UMAP.py` or `classifier.py`