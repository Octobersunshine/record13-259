import requests
import os

BASE_URL = 'http://127.0.0.1:5000'


def step(msg):
    print('\n' + '=' * 60)
    print(msg)
    print('=' * 60)


def upload_sample():
    step('1. 上传 CSV 文件')
    filepath = os.path.join(os.path.dirname(__file__), 'sample_data.csv')
    with open(filepath, 'rb') as f:
        resp = requests.post(
            f'{BASE_URL}/api/upload',
            files={'file': ('sample_data.csv', f, 'text/csv')}
        )
    data = resp.json()
    print(f'状态码: {resp.status_code}')
    print(f'File ID: {data["file_id"]}')
    print(f'原始文件名: {data["original_filename"]}')
    print(f'数据形状: {data["shape"]["rows"]} 行 x {data["shape"]["columns"]} 列')
    print(f'列名: {data["columns"]}')
    return data['file_id']


def get_missing(file_id):
    step('2. 获取缺失值统计')
    resp = requests.get(f'{BASE_URL}/api/missing-stats/{file_id}')
    data = resp.json()
    print(f'状态码: {resp.status_code}')
    print('总体统计:')
    for k, v in data['summary'].items():
        print(f'  {k}: {v}')
    print('\n按列统计:')
    for col, info in data['by_column'].items():
        print(f'  {col}: 缺失 {info["missing_count"]} ({info["missing_percentage"]}%), 类型: {info["dtype"]}')


def drop_dup(file_id):
    step('3. 删除重复行')
    resp = requests.post(f'{BASE_URL}/api/drop-duplicates/{file_id}')
    data = resp.json()
    print(f'状态码: {resp.status_code}')
    print(f'删除重复行数: {data["duplicates_removed"]}')
    print(f'去重前行数: {data["rows_before"]}')
    print(f'去重后行数: {data["rows_after"]}')
    print(f'输出文件: {data["output_file"]}')
    print(f'下载链接: {data["download_url"]}')


def fill_missing(file_id, strategy):
    step(f'4. 填充缺失值 (策略: {strategy})')
    resp = requests.post(
        f'{BASE_URL}/api/fill-missing/{file_id}',
        json={'strategy': strategy}
    )
    data = resp.json()
    print(f'状态码: {resp.status_code}')
    print(f'策略: {data["strategy"]}')
    print('\n填充报告:')
    for col, info in data['fill_report'].items():
        print(f'  {col}: {info}')
    print(f'\n输出文件: {data["output_file"]}')
    print(f'下载链接: {data["download_url"]}')


def clean_pipeline(file_id, strategy='mean'):
    step(f'5. 一键清洗流水线 (策略: {strategy})')
    resp = requests.post(
        f'{BASE_URL}/api/clean/{file_id}',
        json={'strategy': strategy}
    )
    data = resp.json()
    print(f'状态码: {resp.status_code}')
    print(f'填充策略: {data["strategy"]}')
    print(f'删除重复行数: {data["duplicates_removed"]}')
    print(f'清洗前缺失: {data["missing_before"]}')
    print(f'清洗后缺失: {data["missing_after"]}')
    print(f'\n填充报告:')
    for col, info in data['fill_report'].items():
        print(f'  {col}: {info}')
    print(f'\n输出文件: {data["output_file"]}')
    print(f'下载链接: {data["download_url"]}')


if __name__ == '__main__':
    try:
        fid = upload_sample()
        get_missing(fid)
        drop_dup(fid)
        fill_missing(fid, 'mean')
        fill_missing(fid, 'median')
        clean_pipeline(fid, 'mean')
        print('\n所有步骤执行完成!')
    except requests.ConnectionError:
        print('无法连接到服务器，请先运行: python app.py')
