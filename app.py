import os
import uuid
import pandas as pd
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'csv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_csv(filepath):
    return pd.read_csv(filepath)


def save_csv(df, filepath):
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    return filepath


def get_missing_stats(df):
    total_missing = df.isnull().sum().sum()
    total_cells = df.size
    stats = {
        'summary': {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'total_missing_cells': int(total_missing),
            'total_cells': int(total_cells),
            'missing_percentage': round(total_missing / total_cells * 100, 2) if total_cells > 0 else 0
        },
        'by_column': {}
    }
    for col in df.columns:
        missing_count = int(df[col].isnull().sum())
        missing_pct = round(missing_count / len(df) * 100, 2) if len(df) > 0 else 0
        stats['by_column'][col] = {
            'missing_count': missing_count,
            'missing_percentage': missing_pct,
            'dtype': str(df[col].dtype)
        }
    return stats


def drop_duplicates(df):
    before = len(df)
    df_cleaned = df.drop_duplicates()
    after = len(df_cleaned)
    return {
        'dataframe': df_cleaned,
        'duplicates_removed': int(before - after),
        'rows_before': int(before),
        'rows_after': int(after)
    }


def fill_missing(df, strategy='mean'):
    df_filled = df.copy()
    fill_report = {}

    for col in df_filled.columns:
        missing_count = int(df_filled[col].isnull().sum())
        if missing_count == 0:
            fill_report[col] = {'filled': 0, 'method': 'none'}
            continue

        dtype = df_filled[col].dtype

        if pd.api.types.is_numeric_dtype(dtype):
            if strategy == 'mean':
                fill_value = df_filled[col].mean()
            elif strategy == 'median':
                fill_value = df_filled[col].median()
            else:
                fill_report[col] = {'filled': 0, 'method': 'skipped', 'reason': f'unknown strategy: {strategy}'}
                continue
            df_filled[col] = df_filled[col].fillna(fill_value)
            fill_report[col] = {
                'filled': missing_count,
                'method': strategy,
                'fill_value': float(fill_value) if pd.notnull(fill_value) else None
            }
        else:
            mode_values = df_filled[col].mode()
            if len(mode_values) > 0:
                fill_value = mode_values.iloc[0]
                df_filled[col] = df_filled[col].fillna(fill_value)
                fill_report[col] = {
                    'filled': missing_count,
                    'method': 'mode',
                    'note': 'non-numeric column, used mode instead',
                    'fill_value': str(fill_value)
                }
            else:
                fill_report[col] = {'filled': 0, 'method': 'skipped', 'reason': 'all values are null'}

    return {
        'dataframe': df_filled,
        'fill_report': fill_report
    }


def detect_outliers_iqr(df, k=1.5):
    outlier_report = {}
    total_outliers = 0
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col].dtype)]

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            outlier_report[col] = {
                'outlier_count': 0,
                'outlier_percentage': 0.0,
                'q1': None,
                'q3': None,
                'iqr': None,
                'lower_bound': None,
                'upper_bound': None,
                'method': 'iqr',
                'note': 'no valid numeric values'
            }
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr

        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = int(outlier_mask.sum())
        outlier_pct = round(outlier_count / len(df) * 100, 2) if len(df) > 0 else 0.0
        total_outliers += outlier_count

        outlier_values = df.loc[outlier_mask, col].tolist()

        outlier_report[col] = {
            'outlier_count': outlier_count,
            'outlier_percentage': outlier_pct,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'method': 'iqr',
            'k_factor': k,
            'outlier_values': [float(v) for v in outlier_values if pd.notnull(v)]
        }

    non_numeric_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col].dtype)]
    for col in non_numeric_cols:
        outlier_report[col] = {
            'outlier_count': 0,
            'outlier_percentage': 0.0,
            'method': 'skipped',
            'note': 'non-numeric column'
        }

    return {
        'summary': {
            'total_rows': len(df),
            'numeric_columns': len(numeric_cols),
            'total_outlier_cells': total_outliers,
            'method': 'iqr',
            'k_factor': k
        },
        'by_column': outlier_report
    }


def handle_outliers(df, k=1.5, strategy='cap'):
    df_cleaned = df.copy()
    handle_report = {}
    total_handled = 0
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col].dtype)]

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            handle_report[col] = {'handled': 0, 'method': 'skipped', 'note': 'no valid numeric values'}
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr

        outlier_mask = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
        outlier_count = int(outlier_mask.sum())

        if outlier_count == 0:
            handle_report[col] = {'handled': 0, 'method': 'none', 'lower_bound': lower_bound, 'upper_bound': upper_bound}
            continue

        if strategy == 'cap':
            df_cleaned.loc[df_cleaned[col] < lower_bound, col] = lower_bound
            df_cleaned.loc[df_cleaned[col] > upper_bound, col] = upper_bound
            handle_report[col] = {
                'handled': outlier_count,
                'method': 'cap',
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'note': 'values capped at IQR bounds'
            }
        elif strategy == 'remove':
            df_cleaned = df_cleaned[~outlier_mask].reset_index(drop=True)
            handle_report[col] = {
                'handled': outlier_count,
                'method': 'remove',
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'note': 'rows with outliers removed'
            }
        elif strategy == 'median':
            median_val = float(series.median())
            df_cleaned.loc[outlier_mask, col] = median_val
            handle_report[col] = {
                'handled': outlier_count,
                'method': 'median',
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'replacement_value': median_val
            }
        else:
            handle_report[col] = {'handled': 0, 'method': 'skipped', 'reason': f'unknown strategy: {strategy}'}
            continue

        total_handled += outlier_count

    non_numeric_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col].dtype)]
    for col in non_numeric_cols:
        handle_report[col] = {'handled': 0, 'method': 'skipped', 'note': 'non-numeric column'}

    return {
        'dataframe': df_cleaned,
        'handle_report': handle_report,
        'total_handled': total_handled,
        'rows_after': len(df_cleaned)
    }


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Only CSV files are allowed'}), 400

    filename = secure_filename(file.filename)
    file_id = str(uuid.uuid4())
    stored_name = f"{file_id}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], stored_name)
    file.save(filepath)

    try:
        df = load_csv(filepath)
    except Exception as e:
        os.remove(filepath)
        return jsonify({'error': f'Failed to read CSV: {str(e)}'}), 400

    return jsonify({
        'file_id': file_id,
        'original_filename': filename,
        'stored_filename': stored_name,
        'shape': {
            'rows': len(df),
            'columns': len(df.columns)
        },
        'columns': list(df.columns)
    })


@app.route('/api/missing-stats/<file_id>', methods=['GET'])
def missing_stats(file_id):
    filepath = _find_file(file_id)
    if filepath is None:
        return jsonify({'error': 'File not found'}), 404

    df = load_csv(filepath)
    stats = get_missing_stats(df)
    return jsonify(stats)


@app.route('/api/drop-duplicates/<file_id>', methods=['POST'])
def api_drop_duplicates(file_id):
    filepath = _find_file(file_id)
    if filepath is None:
        return jsonify({'error': 'File not found'}), 404

    df = load_csv(filepath)
    result = drop_duplicates(df)

    output_name = f"{file_id}_dedup.csv"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_name)
    save_csv(result['dataframe'], output_path)

    return jsonify({
        'duplicates_removed': result['duplicates_removed'],
        'rows_before': result['rows_before'],
        'rows_after': result['rows_after'],
        'output_file': output_name,
        'download_url': f'/api/download/{output_name}'
    })


@app.route('/api/fill-missing/<file_id>', methods=['POST'])
def api_fill_missing(file_id):
    filepath = _find_file(file_id)
    if filepath is None:
        return jsonify({'error': 'File not found'}), 404

    data = request.get_json(silent=True) or {}
    strategy = data.get('strategy', 'mean')

    if strategy not in ('mean', 'median'):
        return jsonify({'error': 'Invalid strategy. Must be "mean" or "median"'}), 400

    df = load_csv(filepath)
    result = fill_missing(df, strategy=strategy)

    output_name = f"{file_id}_filled_{strategy}.csv"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_name)
    save_csv(result['dataframe'], output_path)

    return jsonify({
        'strategy': strategy,
        'fill_report': result['fill_report'],
        'output_file': output_name,
        'download_url': f'/api/download/{output_name}'
    })


@app.route('/api/detect-outliers/<file_id>', methods=['GET'])
def api_detect_outliers(file_id):
    filepath = _find_file(file_id)
    if filepath is None:
        return jsonify({'error': 'File not found'}), 404

    k = float(request.args.get('k', 1.5))

    df = load_csv(filepath)
    result = detect_outliers_iqr(df, k=k)
    return jsonify(result)


@app.route('/api/handle-outliers/<file_id>', methods=['POST'])
def api_handle_outliers(file_id):
    filepath = _find_file(file_id)
    if filepath is None:
        return jsonify({'error': 'File not found'}), 404

    data = request.get_json(silent=True) or {}
    k = float(data.get('k', 1.5))
    strategy = data.get('strategy', 'cap')

    if strategy not in ('cap', 'remove', 'median'):
        return jsonify({'error': 'Invalid strategy. Must be "cap", "remove", or "median"'}), 400

    df = load_csv(filepath)
    result = handle_outliers(df, k=k, strategy=strategy)

    output_name = f"{file_id}_outliers_{strategy}.csv"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_name)
    save_csv(result['dataframe'], output_path)

    return jsonify({
        'method': 'iqr',
        'k_factor': k,
        'strategy': strategy,
        'total_handled': result['total_handled'],
        'rows_after': result['rows_after'],
        'handle_report': result['handle_report'],
        'output_file': output_name,
        'download_url': f'/api/download/{output_name}'
    })


@app.route('/api/clean/<file_id>', methods=['POST'])
def clean_pipeline(file_id):
    filepath = _find_file(file_id)
    if filepath is None:
        return jsonify({'error': 'File not found'}), 404

    data = request.get_json(silent=True) or {}
    strategy = data.get('strategy', 'mean')
    handle_outliers_flag = data.get('handle_outliers', True)
    outlier_strategy = data.get('outlier_strategy', 'cap')
    outlier_k = float(data.get('outlier_k', 1.5))

    if strategy not in ('mean', 'median'):
        return jsonify({'error': 'Invalid strategy. Must be "mean" or "median"'}), 400

    df = load_csv(filepath)

    missing_stats_before = get_missing_stats(df)
    outliers_before = detect_outliers_iqr(df, k=outlier_k) if handle_outliers_flag else None

    dedup_result = drop_duplicates(df)
    df = dedup_result['dataframe']

    outlier_result = None
    if handle_outliers_flag:
        outlier_result = handle_outliers(df, k=outlier_k, strategy=outlier_strategy)
        df = outlier_result['dataframe']

    fill_result = fill_missing(df, strategy=strategy)
    df = fill_result['dataframe']

    missing_stats_after = get_missing_stats(df)
    outliers_after = detect_outliers_iqr(df, k=outlier_k) if handle_outliers_flag else None

    output_name = f"{file_id}_cleaned_{strategy}.csv"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_name)
    save_csv(df, output_path)

    response = {
        'strategy': strategy,
        'duplicates_removed': dedup_result['duplicates_removed'],
        'missing_before': missing_stats_before['summary'],
        'missing_after': missing_stats_after['summary'],
        'fill_report': fill_result['fill_report'],
        'output_file': output_name,
        'download_url': f'/api/download/{output_name}'
    }

    if handle_outliers_flag:
        response['outliers'] = {
            'method': 'iqr',
            'k_factor': outlier_k,
            'strategy': outlier_strategy,
            'total_outliers_before': outliers_before['summary']['total_outlier_cells'],
            'total_outliers_after': outliers_after['summary']['total_outlier_cells'],
            'handle_report': outlier_result['handle_report'] if outlier_result else {}
        }

    return jsonify(response)


@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], secure_filename(filename))
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    return send_file(filepath, as_attachment=True, download_name=filename)


def _find_file(file_id):
    for fname in os.listdir(app.config['UPLOAD_FOLDER']):
        if fname.startswith(file_id + '_'):
            return os.path.join(app.config['UPLOAD_FOLDER'], fname)
    return None


if __name__ == '__main__':
    app.run(debug=True, port=5000)
