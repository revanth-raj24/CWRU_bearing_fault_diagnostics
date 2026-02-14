# Raw CWRU Bearing Data

Place CWRU bearing vibration `.mat` files here, **one folder per class**:

| Folder        | Label | Description        |
|---------------|-------|--------------------|
| `normal/`     | 0     | Healthy baseline   |
| `inner_race/` | 1     | Inner race fault   |
| `ball/`       | 2     | Ball fault         |

Each `.mat` file must contain a drive-end time signal (e.g. key `DE_time`).  
The pipeline uses the **folder name** to assign labels (see `src/config.py` → `LABEL_MAP`).

## Where to get the data

- **CWRU Bearing Data Center:**  
  https://engineering.case.edu/bearingdatacenter/download

- Download the dataset, then copy `.mat` files into the matching folders above (e.g. normal condition files → `normal/`, inner race fault files → `inner_race/`, ball fault files → `ball/`).

After adding files, run from project root:

```bash
python build_data.py
```
