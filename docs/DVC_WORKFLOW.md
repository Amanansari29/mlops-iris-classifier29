# DVC Workflow Documentation

## Objective

To track dataset versions using DVC and integrate dataset versioning with Git for reproducible machine learning experiments.

## DVC Remote Configuration

A local DVC remote was configured to store dataset artifacts outside the Git repository.

```bash
mkdir -p ~/dvc-remote-storage
dvc remote add -d myremote ~/dvc-remote-storage
The remote configuration was stored in .dvc/config and committed to Git.

---

## Dataset Versioning Workflow

For every dataset change, the following workflow was followed:

### 1. Track Dataset with DVC

dvc add data/raw/iris_v1.csv
This command generated a .dvc metafile containing the dataset hash and metadata.

### 2. Stage DVC Metadata

git add data/raw/iris_v1.csv.dvc data/raw/.gitignore
Only the DVC metadata was staged, not the actual dataset.

### 3. Commit Changes

git commit -m "data: add dataset version"
Git stored the dataset pointer file in version history.

### 4. Push Data to DVC Remote

dvc push
The dataset content was uploaded to the configured DVC remote storage.

---

## Dataset Version Comparison

Dataset versions were compared using:

dvc diff <commit_hash>
This command showed changes between the current dataset version and a previous version.

Example:

dvc diff 602aa4a
Output indicated that data/raw/iris_v1.csv was modified.

---

## Restoring Previous Dataset Versions

### Restore Old DVC Pointer

git checkout <commit_hash> -- data/raw/iris_v1.csv.dvc
This restored the historical DVC metadata file.

### Restore Actual Dataset

dvc checkout data/raw/iris_v1.csv.dvc
DVC used the hash stored in the .dvc file to restore the corresponding dataset version from cache.

### Return to Latest Version

git checkout HEAD -- data/raw/iris_v1.csv.dvc
dvc checkout data/raw/iris_v1.csv.dvc
This restored the latest dataset version.

---

## Conclusion

DVC was successfully integrated with Git to version datasets independently from source code. Dataset versions could be tracked, compared, stored in remote storage, and restored when required, ensuring reproducible machine learning experiments.

```