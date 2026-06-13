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


@app.route('/api/clean/<file_id>', methods=['POST'])
def clean_pipeline(file_id):
    filepath = _find_file(file_id)
    if filepath is None:
        return jsonify({'error': 'File not found'}), 404

    data = request.get_json(silent=True) or {}
    strategy = data.get('strategy', 'mean')

    if strategy not in ('mean', 'median'):
        return jsonify({'error': 'Invalid strategy. Must be "mean" or "median"'}), 400

    df = load_csv(filepath)

    missing_stats_before = get_missing_stats(df)

    dedup_result = drop_duplicates(df)
    df = dedup_result['dataframe']

    fill_result = fill_missing(df, strategy=strategy)
    df = fill_result['dataframe']

    missing_stats_after = get_missing_stats(df)

    output_name = f"{file_id}_cleaned_{strategy}.csv"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_name)
    save_csv(df, output_path)

    return jsonify({
        'strategy': strategy,
        'duplicates_removed': dedup_result['duplicates_removed'],
        'missing_before': missing_stats_before['summary'],
        'missing_after': missing_stats_after['summary'],
        'fill_report': fill_result['fill_report'],
        'output_file': output_name,
        'download_url': f'/api/download/{output_name}'
    })


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
