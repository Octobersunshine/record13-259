import requests
import os

BASE_URL = 'http://127.0.0.1:5000'


def step(msg):
    print('\n' + '=' * 70)
    print(msg)
    print('=' * 70)


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
    print(f'数据形状: {data["shape"]["rows"]} 行 x {data["shape"]["columns"]} 列')
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
        print(f'  {col}: 缺失 {info["missing_count"]} ({info["missing_percentage"]}%)')


def drop_dup(file_id):
    step('3. 删除重复行')
    resp = requests.post(f'{BASE_URL}/api/drop-duplicates/{file_id}')
    data = resp.json()
    print(f'状态码: {resp.status_code}')
    print(f'删除重复行数: {data["duplicates_removed"]}')
    print(f'去重前: {data["rows_before"]} 行, 去重后: {data["rows_after"]} 行')


def detect_outliers(file_id):
    step('4. IQR 异常值检测')
    resp = requests.get(f'{BASE_URL}/api/detect-outliers/{file_id}?k=1.5')
    data = resp.json()
    print(f'状态码: {resp.status_code}')
    print(f'检测方法: {data["summary"]["method"]}')
    print(f'k因子: {data["summary"]["k_factor"]}')
    print(f'异常值单元格总数: {data["summary"]["total_outlier_cells"]}')
    print('\n按列异常值详情:')
    for col, info in data['by_column'].items():
        if info.get('method') == 'iqr':
            print(f'  {col}:')
            print(f'    异常值数量: {info["outlier_count"]} ({info["outlier_percentage"]}%)')
            print(f'    Q1={info["q1"]}, Q3={info["q3"]}, IQR={info["iqr"]}')
            print(f'    下界: {info["lower_bound"]}, 上界: {info["upper_bound"]}')
            print(f'    异常值: {info["outlier_values"]}')
        else:
            print(f'  {col}: {info.get("note", "跳过")}')


def handle_outliers_test(file_id, strategy):
    step(f'5. 异常值处理 (策略: {strategy})')
    resp = requests.post(
        f'{BASE_URL}/api/handle-outliers/{file_id}',
        json={'strategy': strategy, 'k': 1.5}
    )
    data = resp.json()
    print(f'状态码: {resp.status_code}')
    print(f'方法: {data["method"]}, 策略: {data["strategy"]}')
    print(f'处理的异常值总数: {data["total_handled"]}')
    print(f'处理后行数: {data["rows_after"]}')
    print('\n处理报告:')
    for col, info in data['handle_report'].items():
        if info.get('handled', 0) > 0:
            print(f'  {col}: {info}')


def clean_pipeline(file_id, strategy='mean'):
    step(f'6. 一键清洗流水线 (含 IQR 异常值处理)')
    resp = requests.post(
        f'{BASE_URL}/api/clean/{file_id}',
        json={
            'strategy': strategy,
            'handle_outliers': True,
            'outlier_strategy': 'cap',
            'outlier_k': 1.5
        }
    )
    data = resp.json()
    print(f'状态码: {resp.status_code}')
    print(f'填充策略: {data["strategy"]}')
    print(f'删除重复行数: {data["duplicates_removed"]}')
    print(f'缺失值: {data["missing_before"]["total_missing_cells"]} → {data["missing_after"]["total_missing_cells"]}')

    if 'outliers' in data:
        print(f'\n异常值处理 (方法: {data["outliers"]["method"]}):')
        print(f'  异常值数量: {data["outliers"]["total_outliers_before"]} → {data["outliers"]["total_outliers_after"]}')
        print(f'  策略: {data["outliers"]["strategy"]}')

    print(f'\n输出文件: {data["output_file"]}')
    print(f'下载链接: {data["download_url"]}')


def compare_3sigma_vs_iqr(file_id):
    step('7. 验证: 为什么 IQR 比 3σ 更鲁棒 (非正态分布场景)')
    print('''
    ╔═══════════════════════════════════════════════════════════════╗
    ║  IQR 方法 vs 3σ 方法对比                                      ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  3σ 方法 (Z-Score):                                           ║
    ║    - 假设数据服从正态分布                                      ║
    ║    - 基于均值和标准差                                          ║
    ║    - 对偏态分布、有极端值的数据会产生大量误判                    ║
    ║    - 异常值本身会拉高标准差，导致漏检                           ║
    ║                                                              ║
    ║  IQR 方法 (四分位距):                                         ║
    ║    - 不假设数据分布形态，适用于任何分布                         ║
    ║    - 基于分位数 (Q1, Q3)，不受极端值影响                       ║
    ║    - 对偏态数据、长尾分布更鲁棒                                ║
    ║    - 异常值不会影响检测阈值，结果更可靠                         ║
    ╚═══════════════════════════════════════════════════════════════╝
    ''')
    print('本服务已弃用 3σ 方法，全部改用 IQR 方法检测异常值。')
    print('当前测试数据包含: Kate(age=200, salary=500000), Leo(age=2, salary=1000)')
    print('IQR 方法能准确识别这些异常值，不受数据偏态影响。')


if __name__ == '__main__':
    try:
        fid = upload_sample()
        get_missing(fid)
        drop_dup(fid)
        detect_outliers(fid)
        handle_outliers_test(fid, 'cap')
        handle_outliers_test(fid, 'median')
        clean_pipeline(fid, 'mean')
        compare_3sigma_vs_iqr(fid)
        print('\n所有测试执行完成!')
    except requests.ConnectionError:
        print('无法连接到服务器，请先运行: python app.py')
