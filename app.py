# app_flask.py
from flask import Flask, render_template, request, flash, redirect, url_for
import os
from train_model import train_and_eval, prepare_user_input, predict_user

app = Flask(__name__)
app.secret_key = "replace_with_a_random_secret_key"

PLOT_DIR = os.path.join('static', 'plots')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/simulate/<status>')
def simulate(status):
    # contoh data pelanggan (dummy)
    if status == 'churn':
        sample = {
            'gender': 'Female',
            'SeniorCitizen': 1,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 2,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'Yes',
            'StreamingMovies': 'Yes',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 95.5,
            'TotalCharges': 188.2
        }
    else:  # simulate non-churn
        sample = {
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'Yes',
            'tenure': 48,
            'PhoneService': 'Yes',
            'MultipleLines': 'Yes',
            'InternetService': 'DSL',
            'OnlineSecurity': 'Yes',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'Yes',
            'TechSupport': 'Yes',
            'StreamingTV': 'Yes',
            'StreamingMovies': 'Yes',
            'Contract': 'Two year',
            'PaperlessBilling': 'No',
            'PaymentMethod': 'Bank transfer (automatic)',
            'MonthlyCharges': 55.1,
            'TotalCharges': 2640.7
        }

    try:
        X_ready = prepare_user_input(sample)
        result = predict_user(X_ready)
        return render_template('result.html', pred_result=result)
    except FileNotFoundError:
        return render_template('result.html', pred_result={'error': 'Model belum tersedia, lakukan training dulu!'})


@app.route('/train', methods=['POST'])
def train_route():
    try:
        # simple train without tuning
        results = train_and_eval(do_tuning=False)
        flash("Training selesai ✔️")
        return render_template('result.html', results=results, pred_result=None)
    except Exception as e:
        flash(f"Error saat training: {e}")
        return render_template('result.html', results=None, pred_result=None)


@app.route('/tune', methods=['POST'])
def tune_route():
    try:
        # run GridSearch tuning + training
        results = train_and_eval(do_tuning=True)
        flash("Tuning & Training selesai ✔️")
        return render_template('result.html', results=results, pred_result=None)
    except Exception as e:
        flash(f"Error saat tuning: {e}")
        return render_template('result.html', results=None, pred_result=None)


@app.route('/predict', methods=['POST'])
def predict_route():
    input_data = {k: request.form.get(k) for k in request.form}
    try:
        X_ready = prepare_user_input(input_data)
        pred = predict_user(X_ready)
        # try load saved plot fragments if exist
        results = {}
        cm_html = os.path.join(PLOT_DIR, 'cm_plot.html')
        roc_html = os.path.join(PLOT_DIR, 'roc_plot.html')
        feat_html = os.path.join(PLOT_DIR, 'feat_plot.html')
        tree_over = os.path.join(PLOT_DIR, 'decision_tree_overview.png')
        if os.path.exists(cm_html):
            with open(cm_html, 'r', encoding='utf-8') as f:
                results['cm_div'] = f.read()
        if os.path.exists(roc_html):
            with open(roc_html, 'r', encoding='utf-8') as f:
                results['roc_div'] = f.read()
        if os.path.exists(feat_html):
            with open(feat_html, 'r', encoding='utf-8') as f:
                results['feat_div'] = f.read()
        if os.path.exists(tree_over):
            results['tree_path'] = tree_over
        # load tuning csv if exist
        tuning_csv = os.path.join('static', 'tuning_results.csv')
        if os.path.exists(tuning_csv):
            results['tuning_csv'] = tuning_csv
        return render_template('result.html', results=results if results else None, pred_result=pred)
    except FileNotFoundError:
        return render_template('result.html', results=None, pred_result={'error': 'Model belum tersedia. Lakukan training atau tuning dulu.'})
    except Exception as e:
        return render_template('result.html', results=None, pred_result={'error': str(e)})


if __name__ == '__main__':
    os.makedirs(PLOT_DIR, exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
