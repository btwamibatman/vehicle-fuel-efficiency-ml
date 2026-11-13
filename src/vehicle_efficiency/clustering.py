"""Training-only K-Means exploration of engine characteristics."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .data import PROJECT_ROOT, RANDOM_SEED, load_dataset, make_splits

FEATURES = ['cylinders', 'displacement']
RESULTS = PROJECT_ROOT / 'reports/results/endterm'
FIGURES = PROJECT_ROOT / 'reports/figures/endterm'

def build_clusterer(k):
    return Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler()),
                     ('model', KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=20))])

def search(train):
    """Fit only on features, never target; sample for reproducible silhouette."""
    X = train[FEATURES]
    results = []
    for k in range(2, 9):
        model = build_clusterer(k).fit(X)
        z = model[:-1].transform(X)
        labels = model['model'].labels_
        results.append({'k': k, 'inertia': model['model'].inertia_,
                        'silhouette': silhouette_score(z, labels, sample_size=min(3000, len(X)), random_state=RANDOM_SEED),
                        'min_cluster_size': int(np.bincount(labels).min())})
    return pd.DataFrame(results)

def run():
    import matplotlib.pyplot as plt
    from .endterm_data import verified_splits
    splits, manifest = verified_splits()
    train = splits['train']; RESULTS.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    scores = search(train); scores.to_csv(RESULTS / 'clustering_search.csv', index=False)
    # Three groups offer a compact interpretable partition; see notebook for evidence.
    k = 3
    pipe = build_clusterer(k).fit(train[FEATURES]); labels = pipe['model'].labels_
    pd.DataFrame({'row_id': train.row_id.to_numpy(), 'cluster': labels}).to_csv(RESULTS / 'clustering_assignments_train.csv', index=False)
    frame = train.assign(cluster=labels)
    profiles = frame.groupby('cluster').agg(n=('mpg','size'), cylinders_mean=('cylinders','mean'), displacement_mean_l=('displacement','mean'), mpg_mean=('mpg','mean'), mpg_median=('mpg','median'))
    centers = pipe['scale'].inverse_transform(pipe['model'].cluster_centers_)
    profiles['center_cylinders'] = centers[:,0]; profiles['center_displacement_l'] = centers[:,1]
    profiles.to_csv(RESULTS / 'clustering_profiles.csv')
    config = {'features': FEATURES, 'target_used_in_fit_or_selection': False, 'imputation': 'median', 'scaling': 'StandardScaler', 'k': k, 'seed': RANDOM_SEED, 'n_init': 20, 'silhouette_sample_size': 3000, 'dataset_sha256': manifest['raw_sha256'], 'split_assignments_sha256': manifest['split_assignments_sha256'], 'split': 'train'}
    (RESULTS / 'clustering_config.json').write_text(json.dumps(config, indent=2)+'\n')
    for column, label in [('inertia','Within-cluster sum of squares'), ('silhouette','Silhouette score')]:
        fig, ax = plt.subplots(); ax.plot(scores.k, scores[column], marker='o'); ax.set(xlabel='Number of clusters (k)', ylabel=label, title='Training data: K-Means'); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIGURES / f'clustering_{column}.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(); scatter=ax.scatter(frame.cylinders, frame.displacement, c=labels, s=9, alpha=.35, cmap='viridis'); ax.set(xlabel='Cylinders', ylabel='Engine displacement (L)', title='Training engine characteristics by cluster'); fig.colorbar(scatter, ax=ax, label='Cluster ID'); fig.tight_layout(); fig.savefig(FIGURES/'clustering_scatter.png',dpi=150); plt.close(fig)
    print(scores.to_string(index=False)); print(profiles.to_string())
if __name__ == '__main__': run()
