"""Training CV and validation comparison; separately evaluate frozen models on the previously used holdout."""
from __future__ import annotations
import argparse, json, platform, importlib.metadata
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold
from .data import PROJECT_ROOT, RANDOM_SEED, _duplicate_groups, load_dataset, make_splits
from .evaluation import compute_metrics
from .models import build_baseline, build_linear_regression, build_decision_tree, build_knn
from .preprocessing import split_xy
from .endterm_data import verified_splits

OUT = PROJECT_ROOT / 'reports/results/endterm'
FIG = PROJECT_ROOT / 'reports/figures/endterm'
SPECS = {
    'Baseline (mean)': (lambda: build_baseline(), {}),
    'Linear Regression': (lambda: build_linear_regression(engineered=True), {}),
    'Ridge': (lambda: build_linear_regression(engineered=True, alpha=1), {'model__alpha': [.01,.1,1.,10.]}),
    'Decision Tree': (lambda: build_decision_tree(engineered=True), {'model__max_depth': [12,16,None], 'model__min_samples_leaf':[1,2,5]}),
    'KNN': (lambda: build_knn(engineered=True), {'model__n_neighbors':[2,3,4,5], 'model__weights':['uniform','distance'], 'model__p':[1,2]}),
}

def verified_data():
    return verified_splits()

def run_validation():
    import matplotlib.pyplot as plt
    if (OUT/'model_comparison_test.csv').exists():
        raise RuntimeError('Holdout already evaluated; do not reselect models based on its results')
    splits, manifest=verified_data(); train,val=splits['train'],splits['val']; X,y=split_xy(train); Xv,yv=split_xy(val)
    groups=_duplicate_groups(train); models={}; rows=[]; tuning=[]; predictions=[]; params={}
    for name,(builder,grid) in SPECS.items():
        search=GridSearchCV(builder(),grid,scoring='neg_mean_absolute_error',cv=GroupKFold(5),n_jobs=1,return_train_score=True)
        search.fit(X,y,groups=groups); model=search.best_estimator_; models[name]=model
        params[name]=search.best_params_
        for i, p in enumerate(search.cv_results_['params']):
            tuning.append({'model':name,'parameters':json.dumps(p,sort_keys=True),'cv_mae':-search.cv_results_['mean_test_score'][i], 'cv_std':search.cv_results_['std_test_score'][i], 'cv_train_mae':-search.cv_results_['mean_train_score'][i]})
        pred=model.predict(Xv); m=compute_metrics(yv,pred); t=compute_metrics(y,model.predict(X))
        rows.append({'model':name,'selected_parameters':json.dumps(search.best_params_,sort_keys=True),'train_mae':t['mae'],'validation_mae':m['mae'],'validation_rmse':m['rmse'],'validation_r2':m['r2']})
        predictions.append(pd.DataFrame({'model':name,'row_id':val.row_id.to_numpy(),'y_true':yv.to_numpy(),'y_pred':pred}))
        print(name, m, search.best_params_, flush=True)
    from .neural_network import load_ann
    ann=load_ann(expected_manifest=manifest); pred=ann.predict(Xv); saved=pd.read_csv(OUT/'ann_predictions_validation.csv')
    actual=pd.DataFrame({'row_id':val.row_id.to_numpy(),'y_true':yv.to_numpy(),'y_pred':pred})
    merged=actual.merge(saved,on='row_id',validate='one_to_one',suffixes=('_run','_saved'))
    if len(merged)!=len(val) or not np.array_equal(merged.y_true_run,merged.y_true_saved) or not np.allclose(merged.y_pred_run,merged.y_pred_saved,rtol=1e-5,atol=1e-4):
        raise ValueError('ANN predictions, row IDs or targets do not match saved artifacts')
    m=compute_metrics(yv,pred); t=compute_metrics(y,ann.predict(X))
    rows.append({'model':'Feedforward ANN','selected_parameters':json.dumps({'hidden_sizes':list(ann.config.hidden_sizes),'learning_rate':ann.config.learning_rate,'best_epoch':ann.best_epoch}), 'train_mae':t['mae'],'validation_mae':m['mae'],'validation_rmse':m['rmse'],'validation_r2':m['r2']})
    predictions.append(actual.assign(model='Feedforward ANN')[['model','row_id','y_true','y_pred']]); comparison=pd.DataFrame(rows).sort_values('validation_mae')
    comparison.to_csv(OUT/'model_comparison_validation.csv',index=False); pd.DataFrame(tuning).to_csv(OUT/'supervised_tuning.csv',index=False)
    allpred=pd.concat(predictions,ignore_index=True); allpred.to_csv(OUT/'validation_predictions.csv',index=False)
    classic=comparison[comparison.model!='Feedforward ANN'].iloc[0]['model']; winner=comparison.iloc[0]['model']
    errors=[]; largest=[]
    for name in [classic,'Feedforward ANN']:
        p=allpred[allpred.model==name].merge(val[['row_id','make','model','model_year','vehicle_class','hybrid','cylinders','displacement']],on='row_id',validate='one_to_one',suffixes=('_predictor','_vehicle'))
        p['error']=p.y_pred-p.y_true; p['absolute_error']=p.error.abs()
        for col in ['hybrid','vehicle_class']:
            g=p.groupby(col,dropna=False).agg(n=('error','size'),mae=('absolute_error','mean'),mean_error=('error','mean')).reset_index().rename(columns={col:'group'})
            g.insert(0,'group_type',col); g.insert(0,'model',name); errors.append(g)
        largest.append(p.nlargest(10,'absolute_error'))
    pd.concat(errors).to_csv(OUT/'error_by_group_validation.csv',index=False); pd.concat(largest).to_csv(OUT/'largest_errors_validation.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    for axcol,name in enumerate([classic,'Feedforward ANN']):
        p=allpred[allpred.model==name]; ax=axes[0,axcol]; ax.scatter(p.y_true,p.y_pred,s=7,alpha=.3); lim=[min(p.y_true.min(),p.y_pred.min()),max(p.y_true.max(),p.y_pred.max())]; ax.plot(lim,lim,'k--'); ax.set(title=name,xlabel='Actual MPG',ylabel='Predicted MPG')
        ax=axes[1,axcol]; ax.hist(p.y_pred-p.y_true,bins=50); ax.set(xlabel='Residual: predicted - actual (MPG)',ylabel='Count',title=name)
    fig.tight_layout(); fig.savefig(FIG/'evaluation_validation.png',dpi=150); plt.close(fig)
    meta={'dataset':manifest,'classical_parameters':params,'validation_winner':winner,'best_classical':classic,'selection':'lowest validation MAE','units':'MAE and RMSE in US MPG; R2 unitless','ann_selection':'validation early stopping and selection; no ANN CV','holdout_history':'Previously evaluated in midterm; not an independent endterm test','versions': {'python': platform.python_version(), **{name: importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','torch','scipy']}},'test_evaluated':False}
    (OUT/'experiment_metadata.json').write_text(json.dumps(meta,indent=2)+'\n'); print(comparison.to_string(index=False),flush=True)

def run_test():
    if (OUT/'model_comparison_test.csv').exists():
        raise RuntimeError('Holdout already evaluated; preserve the recorded result')
    from .neural_network import load_ann, verify_manifest
    splits,manifest=verified_data(); meta=json.loads((OUT/'experiment_metadata.json').read_text()); verify_manifest(meta['dataset'],manifest)
    train,test=splits['train'],splits['test']; X,y=split_xy(train); Xt,yt=split_xy(test); rows=[]
    for name,(builder,_) in SPECS.items():
        model=builder().set_params(**meta['classical_parameters'][name]); model.fit(X,y); m=compute_metrics(yt,model.predict(Xt)); rows.append({'model':name,**m})
    ann=load_ann(expected_manifest=manifest); rows.append({'model':'Feedforward ANN',**compute_metrics(yt,ann.predict(Xt))})
    pd.DataFrame(rows).to_csv(OUT/'model_comparison_test.csv',index=False)
    meta['test_evaluated']=True; (OUT/'experiment_metadata.json').write_text(json.dumps(meta,indent=2)+'\n'); print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--test',action='store_true'); args=p.parse_args()
    (run_test if args.test else run_validation)()
