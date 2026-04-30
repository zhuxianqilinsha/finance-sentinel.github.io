# model.py （不要改任何内容）
import pandas as pd
import numpy as np

# 数据清洗函数
def clean_numeric(x):
    if pd.isna(x):
        return np.nan
    x_str = str(x).strip()
    x_str = x_str.replace('%', '').replace(',', '').replace(' ', '')
    try:
        return float(x_str)
    except ValueError:
        return np.nan

# 模型计算函数（网页调用这个）
def predict_score(input_df):
    cols = ['财政自给率', '债务率', '人均财政收入', '税收收入占比', '土地财政依赖度', '财政支出增长率']
    positive_cols = ['财政自给率', '人均财政收入', '税收收入占比']
    negative_cols = ['债务率', '土地财政依赖度', '财政支出增长率']

    df = input_df.copy()
    for col in cols:
        df[col] = df[col].apply(clean_numeric)

    for col in positive_cols:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    for col in negative_cols:
        df[col] = (df[col].max() - df[col]) / (df[col].max() - df[col].min())

    data = df[cols].values
    data_sum = data.sum(axis=0)
    p = data / data_sum
    k = 1 / np.log(len(df))
    p_log = np.log(p + 1e-9)
    e = -k * np.sum(p * p_log, axis=0)
    g = 1 - e
    w = g / g.sum()

    max_weight = 0.3
    w = np.clip(w, 0, max_weight)
    w = w / w.sum()

    weighted = df[cols].values * w
    pos_ideal = weighted.max(axis=0)
    neg_ideal = weighted.min(axis=0)
    d_pos = np.sqrt(np.sum((weighted - pos_ideal) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((weighted - neg_ideal) ** 2, axis=1))
    score_raw = d_neg / (d_pos + d_neg)
    score_percent = score_raw * 100

    result_df = input_df[['年份', '县名']].copy()
    result_df['综合得分（百分制）'] = score_percent.round(2)
    result_df['排名'] = result_df.groupby('年份')['综合得分（百分制）'].rank(ascending=False, method='first').astype(int)

    # 预警分位数（你要求保留）
    result_df['预警分位数'] = result_df.groupby('年份')['综合得分（百分制）'].transform(lambda x: x.quantile(0.2)).round(2)
    result_df['分位数60%'] = result_df.groupby('年份')['综合得分（百分制）'].transform(lambda x: x.quantile(0.6)).round(2)

    def get_level(row):
        if row['综合得分（百分制）'] <= row['预警分位数']:
            return '红灯（高风险）'
        elif row['综合得分（百分制）'] <= row['分位数60%']:
            return '黄灯（关注）'
        else:
            return '绿灯（健康）'

    result_df['预警等级'] = result_df.apply(get_level, axis=1)
    result_df.drop(['分位数60%'], axis=1, inplace=True)

    return result_df