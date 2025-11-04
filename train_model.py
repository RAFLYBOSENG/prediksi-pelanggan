# train_model.py
import os
import time
import joblib
import pandas as pd
import numpy as np

# Gunakan backend non-interaktif agar tidak memakai GUI/Tkinter saat di server
import matplotlib
matplotlib.use('Agg')

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# paths
PLOT_DIR = os.path.join('static', 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)
MODEL_PATH = os.path.join('static', 'model_rf.joblib')
CSV_PATH = os.path.join('data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
TUNING_CSV = os.path.join('static', 'tuning_results.csv')

# helpers: plotly visualizations


def _plotly_confusion_matrix(cm, class_names):
    z = cm.tolist()
    # siapkan teks angka untuk tiap sel
    text = [[str(v) for v in row] for row in cm]
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=class_names,
        y=class_names,
        hoverinfo='skip',
        colorscale='Blues',
        showscale=False,
        text=text,
        texttemplate='%{text}',
        textfont={"color": "black", "size": 14}
    ))
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        margin=dict(t=40, l=20, r=20, b=20)
    )
    return fig


def _plotly_roc(fpr, tpr, thresholds, roc_auc):
    fig = go.Figure()
    # siapkan customdata untuk menampilkan threshold pada hover
    customdata = np.array(thresholds)
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name='ROC',
        line=dict(color='#1f77b4', width=3),
        customdata=customdata,
        hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<br>Threshold: %{customdata:.3f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Acak',
        line=dict(dash='dash', color='#888'),
        hoverinfo='skip'
    ))
    # titik operasi default threshold 0.5 (jika ada yang paling dekat)
    try:
        idx = int(np.argmin(np.abs(customdata - 0.5)))
        fig.add_trace(go.Scatter(
            x=[fpr[idx]], y=[tpr[idx]], mode='markers', name='Threshold ~ 0.5',
            marker=dict(color='#d62728', size=10),
            hovertemplate='Operating Point<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}<br>Threshold~0.5<extra></extra>'
        ))
    except Exception:
        pass

    fig.add_annotation(
        x=0.72, y=0.1, xref='paper', yref='paper',
        text=f"AUC = {roc_auc:.4f}",
        showarrow=False,
        font=dict(size=14, color='white'),
        align='center',
        bgcolor='#1f77b4',
        bordercolor='#0d3b66',
        borderwidth=1,
        opacity=0.9
    )
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis=dict(range=[0, 1], constrain='domain', zeroline=True, zerolinecolor='#ccc'),
        yaxis=dict(range=[0, 1], scaleanchor='x', scaleratio=1, zeroline=True, zerolinecolor='#ccc'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(t=50, l=40, r=20, b=40),
        template='plotly_white'
    )
    return fig


def _plotly_feature_importance(importances, feature_names, top_n=10):
    idx = np.argsort(importances)[::-1][:top_n]
    fig = go.Figure(go.Bar(x=importances[idx], y=np.array(
        feature_names)[idx], orientation='h'))
    fig.update_layout(title=f'Top {top_n} Feature Importances', xaxis_title='Importance', margin=dict(
        t=40, l=120, r=20, b=20), yaxis=dict(autorange='reversed'))
    return fig

# data loader


def load_churn_data(csv_path=CSV_PATH):
    # fallback path jika tidak ditemukan di folder data/
    if not os.path.exists(csv_path):
        alt_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
        if os.path.exists(alt_path):
            csv_path = alt_path
        else:
            raise FileNotFoundError(
                f"Dataset tidak ditemukan di {csv_path}. Letakkan file CSV di folder data/ atau di root proyek.")
    df = pd.read_csv(csv_path)
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    encoders = {}
    # Encode semua kolom kategori KECUALI target
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'Churn':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    # Encode target 'Churn' manual agar konsisten: Yes=1, No=0
    if 'Churn' in df.columns:
        # Normalisasi string (strip spasi, title-case)
        df['Churn'] = df['Churn'].astype(str).str.strip()
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0, '1': 1, '0': 0})
        # Buang baris yang Churn-nya tidak valid/NaN
        df = df[df['Churn'].isin([0, 1])].copy()
        df['Churn'] = df['Churn'].astype(int)
    else:
        raise ValueError("Kolom target 'Churn' tidak ditemukan di dataset.")

    # Pastikan tidak ada NaN tersisa di fitur
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df, encoders

# gridsearch tuning


def tune_hyperparameters(X_train, y_train, param_grid=None, cv=3):
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid = GridSearchCV(rf, param_grid, cv=cv, scoring='roc_auc',
                        n_jobs=-1, return_train_score=True)
    grid.fit(X_train, y_train)
    # select small set of columns
    df_res = pd.DataFrame(grid.cv_results_)
    cols = [c for c in df_res.columns if c.startswith(
        'param_')] + ['mean_test_score', 'rank_test_score']
    df_show = df_res[cols].sort_values(
        'rank_test_score').reset_index(drop=True)
    # save CSV of tuning results (for history)
    df_show.to_csv(TUNING_CSV, index=False)
    return grid, df_show

# main train & eval (with optional tuning)


def train_and_eval(csv_path=CSV_PATH, do_tuning=True):
    start = time.time()
    df, encoders = load_churn_data(csv_path)
    if 'Churn' not in df.columns:
        raise ValueError("Kolom target 'Churn' tidak ditemukan di dataset.")

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    best_estimator = None
    tuning_table = None
    best_score = None
    tuning_time = 0.0

    if do_tuning:
        try:
            t0 = time.time()
            grid, tuning_df = tune_hyperparameters(X_train, y_train)
            tuning_time = time.time() - t0
            best_estimator = grid.best_estimator_
            best_score = grid.best_score_
            # convert df to records to be easily displayed in template
            tuning_table = tuning_df.to_dict(orient='records')
        except Exception as e:
            # jika tuning gagal, fallback ke default model
            print("Tuning error:", e)
            best_estimator = None
            tuning_table = None

    # if no best_estimator from tuning, train default RF
    train_fit_time = 0.0
    if best_estimator is None:
        model = RandomForestClassifier(
            n_estimators=200, random_state=42, oob_score=True, n_jobs=-1)
        t1 = time.time()
        model.fit(X_train, y_train)
        train_fit_time = time.time() - t1
    else:
        # build final model using best params but enforce oob_score=True & bootstrap=True
        try:
            best_params = best_estimator.get_params()
            best_params.update({'oob_score': True, 'bootstrap': True,
                                'random_state': 42, 'n_jobs': -1})
            model = RandomForestClassifier(**best_params)
        except Exception:
            model = best_estimator
            if hasattr(model, 'set_params'):
                try:
                    model.set_params(oob_score=True, bootstrap=True)
                except Exception:
                    pass
        t1 = time.time()
        # fit final model to compute oob_score_
        model.fit(X_train, y_train)
        train_fit_time = time.time() - t1

    # evaluation on X_test
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.predict(X_test)

    report_text = classification_report(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # interactive plotly divs
    class_names = ['No', 'Yes']
    cm_fig = _plotly_confusion_matrix(cm, class_names)
    roc_fig = _plotly_roc(fpr, tpr, thresholds, roc_auc)
    feat_fig = _plotly_feature_importance(
        model.feature_importances_, X.columns, top_n=10)

    cm_div = pio.to_html(cm_fig, full_html=False, include_plotlyjs='cdn')
    roc_div = pio.to_html(roc_fig, full_html=False, include_plotlyjs=False)
    feat_div = pio.to_html(feat_fig, full_html=False, include_plotlyjs=False)

    # save html fragments (so /predict dapat load kembali tanpa retrain)
    with open(os.path.join(PLOT_DIR, 'cm_plot.html'), 'w', encoding='utf-8') as f:
        f.write(cm_div)
    with open(os.path.join(PLOT_DIR, 'roc_plot.html'), 'w', encoding='utf-8') as f:
        f.write(roc_div)
    with open(os.path.join(PLOT_DIR, 'feat_plot.html'), 'w', encoding='utf-8') as f:
        f.write(feat_div)

    # save a small decision tree overview image (matplotlib)
    try:
        from sklearn.tree import plot_tree
        plt.figure(figsize=(20, 10))
        estimator = model.estimators_[0] if hasattr(
            model, 'estimators_') else model
        plot_tree(estimator, feature_names=X.columns, class_names=[
                  'No', 'Yes'], filled=True, rounded=True, max_depth=3)
        tree_path = os.path.join(PLOT_DIR, 'decision_tree_overview.png')
        plt.title("Decision Tree (estimator[0], max_depth=3)")
        plt.savefig(tree_path, bbox_inches='tight')
        plt.close()
    except Exception:
        tree_path = None

    # persist model+scaler+encoders+feature_names
    joblib.dump({'model': model, 'scaler': scaler, 'encoders': encoders,
                'feature_names': list(X.columns)}, MODEL_PATH)

    results = {
        'report_text': report_text,
        'report_dict': report_dict,
        'roc_auc': roc_auc,
        'oob_score': getattr(model, 'oob_score_', None),
        'train_time': time.time() - start,
        'train_fit_time': train_fit_time,
        'tuning_time': tuning_time,
        'roc_div': roc_div,
        'cm_div': cm_div,
        'feat_div': feat_div,
        'cm_html_path': os.path.join(PLOT_DIR, 'cm_plot.html'),
        'roc_html_path': os.path.join(PLOT_DIR, 'roc_plot.html'),
        'feat_html_path': os.path.join(PLOT_DIR, 'feat_plot.html'),
        'tree_path': tree_path,
        'tuning_table': tuning_table,
        'tuning_csv': TUNING_CSV if os.path.exists(TUNING_CSV) else None,
        'best_tuning_score': best_score
    }
    return results

# functions to prepare single user input and predict


def prepare_user_input(input_dict):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model tidak ditemukan. Jalankan training dulu.")
    modelobj = joblib.load(MODEL_PATH)
    feature_names = modelobj.get('feature_names')
    if feature_names is None:
        raise RuntimeError("Feature names tidak tersedia dalam model.")
    # build df with feature_names order
    df = pd.DataFrame([{k: input_dict.get(k, "") for k in feature_names}])
    # numeric conversions
    for col in ['TotalCharges', 'MonthlyCharges', 'tenure']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    encoders = modelobj.get('encoders', {})
    for col in df.columns:
        if col in encoders:
            le = encoders[col]
            try:
                df[col] = le.transform(df[col])
            except Exception:
                # fallback mapping for common labels
                df[col] = df[col].map({'Yes': 1, 'No': 0, 'Female': 0, 'Male': 1,
                                      'DSL': 0, 'Fiber optic': 1, 'No': 2}).fillna(0).astype(int)
        elif col == "Churn":
            # target column special case, map manually for prediction
            df[col] = df[col].map({'Yes': 1, 'No': 0})
        else:
            # attempt generic mapping for string columns
            if df[col].dtype == object:
                df[col] = df[col].map(
                    {'Yes': 1, 'No': 0, 'Female': 0, 'Male': 1, 'DSL': 0, 'Fiber optic': 1, 'No': 2}).fillna(0)
    df.fillna(0, inplace=True)
    scaler = modelobj['scaler']
    X_scaled = scaler.transform(df[feature_names])
    return X_scaled


def plot_decision_path_for_user(X_ready):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model tidak ditemukan. Jalankan training dulu.")
    modelobj = joblib.load(MODEL_PATH)
    model = modelobj['model']
    estimator = model.estimators_[0] if hasattr(
        model, 'estimators_') else model
    node_indicator = estimator.decision_path(X_ready)
    leaf_id = estimator.apply(X_ready)
    node_index = node_indicator.indices[node_indicator.indptr[0]: node_indicator.indptr[1]]
    feature_names = modelobj.get('feature_names', None)
    plt.figure(figsize=(20, 10))
    try:
        from sklearn.tree import plot_tree
        ax = plt.gca()
        plot_tree(estimator, filled=True, rounded=True, feature_names=feature_names,
                  class_names=['No', 'Yes'], max_depth=3, ax=ax)
    except Exception:
        pass
    path_str = f"Node dilalui: {list(node_index)}   |   Leaf id: {leaf_id[0]}"
    plt.figtext(0.01, 0.01, path_str, fontsize=14, bbox={
                'facecolor': 'yellow', 'alpha': 0.5, 'pad': 5})
    save_path = os.path.join(PLOT_DIR, 'decision_path_user.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    # return path relative to app root for template usage
    return save_path.replace("\\", "/")


def predict_user(X_ready):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model tidak ditemukan. Jalankan training dulu.")
    modelobj = joblib.load(MODEL_PATH)
    model = modelobj['model']
    pred = int(model.predict(X_ready)[0])
    prob = float(model.predict_proba(X_ready)[0][1]) if hasattr(
        model, 'predict_proba') else float(model.predict(X_ready)[0])
    img_path = plot_decision_path_for_user(X_ready)
    # return relative path for HTML
    return {'churn_pred': pred, 'prob_churn': prob, 'decision_path': img_path.replace("\\", "/")}


# for debug
if __name__ == '__main__':
    res = train_and_eval()
    print(res['report_text'])
    print("AUC:", res['roc_auc'])
