import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from PIL import Image
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

st.set_page_config(page_title="县级财政健康智能检测平台", page_icon="📊", layout="wide")

# ------------------ 辅助函数：安全转数值 ------------------
def to_numeric_series(s):
    return pd.to_numeric(s.astype(str).str.replace('%', '').str.replace(',', '').str.strip(), errors='coerce')

# ------------------ 数据加载（6指标完整版 + 自动列名映射） ------------------
@st.cache_data
def load_data():
    indicators_path = r"C:\Users\22652\Desktop\财政哨兵\30个县域2020-2024年财政6个指标数据.csv"
    score_path = r"C:\Users\22652\Desktop\财政哨兵\财政得分矩阵_带三等级预警_含分位数.csv"

    def read_with_encoding(path):
        for enc in ['gbk', 'gb2312', 'utf-8-sig', 'utf-8']:
            try:
                df = pd.read_csv(path, encoding=enc)
                df.columns = [c.strip() for c in df.columns]
                return df
            except:
                continue
        raise ValueError(f"无法读取文件: {path}")

    df_indicators = read_with_encoding(indicators_path)
    df_score = read_with_encoding(score_path)

    standard_indicators = [
        '财政自给率', '债务率', '人均财政收入',
        '税收收入占比', '土地财政依赖度', '财政支出增长率'
    ]

    indicator_mapping = {
        '财政自给率': '财政自给率', '债务率': '债务率', '人均财政收入': '人均财政收入',
        '人均财政收入（元）': '人均财政收入', '税收收入占比': '税收收入占比',
        '税收占比': '税收收入占比', '税收收入比': '税收收入占比',
        '土地财政依赖度': '土地财政依赖度', '土地依赖度': '土地财政依赖度',
        '土地财政依赖度（%）': '土地财政依赖度', '财政支出增长率': '财政支出增长率',
        '支出增长率': '财政支出增长率'
    }

    def normalize_columns(df):
        rename_dict = {}
        for col in df.columns:
            for src, target in indicator_mapping.items():
                if src == col or src in col:
                    rename_dict[col] = target
                    break
        if rename_dict:
            df.rename(columns=rename_dict, inplace=True)
        for ind in standard_indicators:
            if ind not in df.columns:
                df[ind] = np.nan
        return df

    df_indicators = normalize_columns(df_indicators)
    df_score = normalize_columns(df_score)

    for ind in standard_indicators:
        if ind in df_indicators.columns:
            df_indicators[ind] = to_numeric_series(df_indicators[ind])
        if ind in df_score.columns:
            df_score[ind] = to_numeric_series(df_score[ind])

    if '综合得分' in df_score.columns:
        df_score['综合得分'] = to_numeric_series(df_score['综合得分'])
    else:
        df_score.rename(columns={df_score.columns[2]: '综合得分'}, inplace=True)
        df_score['综合得分'] = to_numeric_series(df_score['综合得分'])

    df = pd.merge(df_score, df_indicators, on=['县名', '年份'], how='left', suffixes=('', '_y'))
    for ind in standard_indicators:
        if f"{ind}_y" in df.columns:
            df[ind] = df[ind].combine_first(df[f"{ind}_y"])
            df.drop(columns=[f"{ind}_y"], inplace=True)

    for ind in standard_indicators:
        if ind not in df.columns:
            df[ind] = np.nan
        else:
            df[ind] = pd.to_numeric(df[ind], errors='coerce')

    df['年份'] = df['年份'].astype(int)

    if '排名' not in df.columns:
        df['排名'] = df.groupby('年份')['综合得分'].rank(ascending=False, method='min').astype(int)
    df['排名文字'] = df['排名'].apply(lambda x: f"{x}/30")

    def get_level(row, grp):
        total = len(grp)
        pct = (row['排名'] - 1) / total
        if pct < 0.4:
            return '绿灯（健康）'
        elif pct < 0.8:
            return '黄灯（关注）'
        else:
            return '红灯（高风险）'

    df['预警等级'] = df.groupby('年份', group_keys=False).apply(
        lambda g: g.apply(lambda r: get_level(r, g), axis=1)
    )
    icon_map = {'绿灯（健康）': '🟢', '黄灯（关注）': '🟡', '红灯（高风险）': '🔴'}
    df['预警图标'] = df['预警等级'].map(icon_map)

    counties_coords = {
        "昆山": (120.95, 31.39), "江阴": (120.27, 31.91), "义乌": (120.06, 29.31),
        "晋江": (118.56, 24.78), "荣成": (122.42, 37.16), "常熟": (120.74, 31.64),
        "慈溪": (121.24, 30.17), "宜兴": (119.82, 31.36), "长沙县": (113.08, 28.24),
        "新郑": (113.71, 34.36), "肥西": (117.15, 31.72), "大冶": (114.98, 30.10),
        "南昌县": (115.94, 28.56), "浏阳": (113.63, 28.15), "宁乡": (112.56, 28.27),
        "长丰": (117.17, 32.47), "仁怀": (106.40, 27.81), "简阳": (104.53, 30.40),
        "安宁": (102.48, 24.92), "神木": (110.47, 38.81), "平果": (107.58, 23.32),
        "大理": (100.22, 25.61), "盘州": (104.47, 25.71), "瓦房店": (121.97, 39.62),
        "海城": (122.70, 40.88), "公主岭": (124.82, 43.50), "梅河口": (125.65, 42.53),
        "肇东": (125.97, 46.07), "通榆县": (123.05, 44.81), "普兰店": (121.97, 39.39)
    }

    return df, counties_coords

# ------------------ 预警原因生成 ------------------
def generate_warning_reasons(row):
    reasons, suggestions = [], []
    debt = row.get('债务率', 0)
    if pd.notna(debt) and debt > 1.2:
        reasons.append("债务率超过120%警戒线")
        suggestions.append("控制新增债务，盘活存量资产")
    self_suff = row.get('财政自给率', 0)
    if pd.notna(self_suff) and self_suff < 0.4:
        reasons.append("财政自给率低于40%，依赖转移支付")
        suggestions.append("培育税源，提高收入质量")
    land = row.get('土地财政依赖度', 0)
    if pd.notna(land) and land > 0.4:
        reasons.append("土地财政依赖度超过40%，结构风险较高")
        suggestions.append("降低土地财政依赖，发展实体经济")
    spend = row.get('财政支出增长率', 0)
    if pd.notna(spend) and spend > 0.1:
        reasons.append("财政支出增长率超过10%，财政可持续性承压")
        suggestions.append("优化支出结构，严控非必要开支")
    if not reasons:
        reasons.append("主要指标处于健康区间")
        suggestions.append("持续优化财政结构，保持稳健发展")
    return reasons[:3], suggestions[:2]

# ------------------ PDF报告生成 ------------------
def generate_pdf_report(county_name, year, score, rank, warning_level, reasons, suggestions, trend_fig):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"{county_name} {year}年财政健康评估报告", styles['Title']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"综合得分：{score:.1f}分", styles['Heading2']))
    story.append(Paragraph(f"排名：{rank}", styles['BodyText']))
    story.append(Paragraph(f"预警等级：{warning_level}", styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("预警原因：", styles['Heading3']))
    for r in reasons:
        story.append(Paragraph(f"- {r}", styles['BodyText']))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("改进建议：", styles['Heading3']))
    for s in suggestions:
        story.append(Paragraph(f"- {s}", styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    img_buffer = BytesIO()
    trend_fig.write_image(img_buffer, format='png', width=600, height=300)
    img_buffer.seek(0)
    story.append(ReportImage(img_buffer, width=6*inch, height=3*inch))
    doc.build(story)
    buffer.seek(0)
    return buffer

# ------------------ 平台介绍（动态渐变封面，完全保留第一个代码样式） ------------------
def platform_intro():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(-45deg, #0a2a3b, #1a4a6f, #0f2b3d, #1e3c72);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .intro-container {
        background: rgba(15,43,61,0.75);
        backdrop-filter: blur(12px);
        border-radius: 32px;
        padding: 2rem;
        margin: 2rem auto;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3), 0 0 20px rgba(0,255,255,0.3);
        border: 1px solid rgba(255,255,255,0.2);
        animation: fadeInUp 1s ease-out;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .logo-title-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 20px;
        margin-bottom: 1rem;
        animation: glowPulse 2s infinite alternate;
    }
    @keyframes glowPulse {
        from { text-shadow: 0 0 5px rgba(0,255,255,0.5); }
        to { text-shadow: 0 0 20px rgba(0,255,255,0.9); }
    }
    .logo-title-row img {
        width: 80px;
        height: auto;
        filter: drop-shadow(0 0 8px #00ccff);
        transition: transform 0.3s;
    }
    .logo-title-row img:hover { transform: scale(1.05); }
    .logo-title-row h1 {
        font-size: 3rem;
        margin: 0;
        background: linear-gradient(135deg, #fff, #7bc5ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
    }
    .feature-card {
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .feature-card:hover {
        transform: translateY(-8px);
        background: rgba(255,255,255,0.2);
        box-shadow: 0 15px 30px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)
    try:
        logo = Image.open("logo.png")
        buffered = BytesIO()
        logo.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        logo_html = f'<img src="data:image/png;base64,{img_str}" alt="Logo">'
    except:
        logo_html = '<span style="font-size: 4rem;">🛡️</span>'
    st.markdown(f"""
    <div class="intro-container">
        <div class="logo-title-row">{logo_html}<h1>财政哨兵</h1></div>
        <h1 style="text-align:center; font-size:2.5rem; margin:0; color:white;">县级财政健康智能检测平台</h1>
        <p style="text-align:center; font-size:1.2rem; color:#cce7ff;">实时监测 · 风险预警 · 科学研判 · 决策支持</p>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    features = [
        ("📊","智能评分","多维度综合评分，科学排名"),
        ("⚠️","风险预警","红黄绿灯三级预警"),
        ("🗺️","空间演变","风险分布演变"),
        ("📈","对比分析","县域横向对比"),
        ("🔍","短板诊断","指标体系诊断"),
        ("📄","报告导出","一键生成报告")
    ]
    for i,(icon,title,desc) in enumerate(features):
        with [col1,col2,col3][i%3]:
            st.markdown(f'<div class="feature-card" style="background:rgba(255,255,255,0.1); border-radius:20px; padding:1.5rem; text-align:center;"><h2 style="font-size:2.5rem;">{icon}</h2><h3 style="color:white;">{title}</h3><p style="color:#d0e5ff;">{desc}</p></div>', unsafe_allow_html=True)

# ------------------ 使用指南（保留第一个代码的丰富内容） ------------------
def usage_guide():
    st.markdown("## 📖 使用指南")
    with st.expander("🎯 快速入门", expanded=True):
        st.markdown("""
        1. 左侧边栏选择年份和县域
        2. 查看综合评分、风险预警
        3. 使用对比分析、短板诊断、动态地图等功能
        """)
    with st.expander("📊 指标说明"):
        st.markdown("""
        | 指标 | 含义 | 健康标准 |
        |------|------|----------|
        | 财政自给率 | 一般预算收入/一般预算支出 | >60% |
        | 债务率 | 债务余额/综合财力 | <100% |
        | 人均财政收入 | 财政收入/常住人口 | 越高越好 |
        | 税收收入占比 | 税收收入/财政收入 | >70% |
        | 土地财政依赖度 | 土地出让收入/财政收入 | <30% |
        | 财政支出增长率 | 财政支出同比增长率 | 0-8% |
        """)
    with st.expander("⚠️ 预警规则"):
        st.markdown("""
        - 🟢 绿灯(健康): 综合得分排名前40%
        - 🟡 黄灯(关注): 排名40%-80%
        - 🔴 红灯(高风险): 排名后20%
        """)
    with st.expander("💡 高级功能说明"):
        st.markdown("""
        **1. 县域对比分析**
        - 可同时选择多个县域进行横向对比
        - 自动生成雷达图、趋势图、得分对比卡片

        **2. 短板诊断**
        - 自动识别薄弱指标并给出改进建议
        - 标准化得分越低，风险越高

        **3. 风险地图**
        - 支持逐年滑动查看空间演变
        - 绿色=健康，橙色=关注，红色=高风险

        **4. PDF报告**
        - 一键生成完整评估报告
        - 包含得分、排名、预警、建议、趋势图
        """)

    st.success("✅ 指南使用完毕，祝您使用愉快！")
    st.info("💡 如需进一步分析，可在核心功能中切换县域与年份。")

# ------------------ 核心功能（完全采用第二个代码的精华部分，并适配现有数据） ------------------
def core_functions(df, counties_coords):
    counties = sorted(df['县名'].unique())
    years = sorted(df['年份'].unique(), reverse=True)

    with st.sidebar:
        st.markdown("### 🎛️ 控制面板")
        y = st.selectbox("📅 年份", years)
        c = st.selectbox("🏙️ 县域", counties)

        # PDF报告生成（使用第一个代码的完整版，带趋势图）
        if st.button("📄 生成PDF报告", use_container_width=True):
            with st.spinner("生成报告中..."):
                cur = df[(df['年份']==y) & (df['县名']==c)].iloc[0]
                score = cur['综合得分']
                rank = cur['排名文字']
                warning = cur['预警等级']
                reasons, suggests = generate_warning_reasons(cur)
                trend_data = df[df['县名']==c].sort_values('年份')
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=trend_data['年份'], y=trend_data['综合得分'], mode='lines+markers', line=dict(color='#2a5298')))
                fig.update_layout(height=300, template='plotly_white')
                pdf = generate_pdf_report(c, y, score, rank, warning, reasons, suggests, fig)
                st.download_button("📥 下载报告", data=pdf, file_name=f"{c}_{y}年报告.pdf", mime="application/pdf", use_container_width=True)

    # 获取当前选中数据
    cur = df[(df['年份']==y) & (df['县名']==c)]
    if cur.empty:
        st.warning("无数据")
        return
    cur = cur.iloc[0]

    # ------------------ 优化后的超级好看卡片（整合第二个代码样式） ------------------
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="background:#f0f2f6;border-radius:18px;padding:2.2rem;box-shadow:0 4px 12px rgba(0,0,0,0.05);">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1.6rem;">
                <h3 style="margin:0;font-size:2rem;font-weight:600;color:#222;">综合财政健康评分</h3>
            </div>
            <div style="display:flex;align-items:center;margin-bottom:1.8rem;">
                <span style="font-size:5.2rem;font-weight:700;color:#d32f2f;line-height:1;margin-right:2.4rem;">{cur['综合得分']:.2f}</span>
                <div>
                    <span style="background:#e3f2fd;color:#1976d2;padding:0.5rem 1.2rem;border-radius:14px;font-size:1.25rem;font-weight:500;">全县排名 {cur['排名']} 名</span>
                </div>
            </div>
            <div style="width:100%;height:14px;background:#e0e0e0;border-radius:8px;">
                <div style="width:{cur['综合得分']:.0f}%;height:100%;background:#1976d2;border-radius:8px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        level = cur['预警等级']
        icon = cur['预警图标']
        reasons, _ = generate_warning_reasons(cur)
        bg = "#2e7d32" if "绿灯" in level else "#ed6c02" if "黄灯" in level else "#d32f2f"
        st.markdown(f"""
        <div style="background:{bg};border-radius:18px;padding:2.2rem;color:white;box-shadow:0 4px 12px rgba(0,0,0,0.1);">
            <h2 style="margin:0 0 1rem 0;font-size:2rem;">{icon} 风险预警 | {level}</h2>
            <ul style="font-size:1.15rem;line-height:1.8;padding-left:1.2rem;">
                {''.join([f'<li>{x}</li>' for x in reasons])}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ------------------ 历史趋势 ------------------
    st.markdown("---")
    st.subheader("📈 历史趋势分析")
    trend = df[df['县名']==c].sort_values('年份')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend['年份'], y=trend['综合得分'], mode='lines+markers', line_width=3))
    fig.update_layout(height=360, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    # ------------------ 6大指标全维度可视化（第二个代码的精华） ------------------
    st.markdown("---")
    st.subheader("📊 6大指标全维度可视化")
    inds = ['财政自给率','债务率','人均财政收入','税收收入占比','土地财政依赖度','财政支出增长率']
    t1, t2 = st.tabs(["📈 单县历史趋势", "📊 当年全县对比"])

    with t1:
        ct = df[df['县名']==c].sort_values('年份')
        for idx, i in enumerate(inds):
            if i not in df.columns: continue
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ct['年份'], y=ct[i], mode='lines+markers', name=i))
            fig.update_layout(title=i, height=260, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

    with t2:
        yr = df[df['年份']==y].sort_values('综合得分', ascending=False)
        for i in inds:
            if i not in df.columns: continue
            fig = go.Figure(go.Bar(x=yr['县名'], y=yr[i], name=i))
            fig.update_layout(title=i, height=260, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

    # ------------------ 短板诊断 ------------------
    st.markdown("---")
    st.subheader("🔬 指标体系诊断 - 短板定位")
    avail = [i for i in inds if i in df.columns]
    if avail:
        stds = []
        for i in avail:
            v = cur[i]
            if pd.isna(v):
                stds.append(0.5)
                continue
            yv = df[df['年份']==y][i].dropna()
            if len(yv)==0:
                stds.append(0.5)
                continue
            mi, ma = yv.min(), yv.max()
            if ma==mi:
                stds.append(0.5)
            else:
                if i in ['债务率','土地财政依赖度','财政支出增长率']:
                    stds.append(1-(v-mi)/(ma-mi))
                else:
                    stds.append((v-mi)/(ma-mi))
        colors = ['red' if s<0.4 else '#1976d2' for s in stds]
        fig = go.Figure(go.Bar(x=avail, y=stds, marker_color=colors))
        fig.update_layout(height=360, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    # ------------------ 仪表盘（债务率、自给率、土地依赖度） ------------------
    st.markdown("---")
    st.subheader("🎯 核心指标健康仪表盘")
    a,b,c3 = st.columns(3)
    with a:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=cur["债务率"], title={'text':"债务率"}, gauge={'axis':{'range':[0,2]}}))
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=cur["财政自给率"], title={'text':"财政自给率"}, gauge={'axis':{'range':[0,1]}}))
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=cur["土地财政依赖度"], title={'text':"土地依赖度"}, gauge={'axis':{'range':[0,1]}}))
        st.plotly_chart(fig, use_container_width=True)

    # ------------------ 排行榜 ------------------
    st.markdown("---")
    st.subheader("🏅 当年县域综合得分排行榜")
    rk = df[df['年份']==y].sort_values('综合得分', ascending=False)
    st.dataframe(rk[['排名','县名','综合得分','预警等级']], hide_index=True, use_container_width=True)

    # ------------------ 雷达对比 ------------------
    st.markdown("---")
    st.subheader("🔄 多县域指标雷达对比")
    cmp = st.multiselect("选择2-4个对比县域", counties, default=counties[:2])
    if len(cmp)>=2:
        d = df[(df['年份']==y) & (df['县名'].isin(cmp))]
        fig = go.Figure()
        for cy in cmp:
            r = d[d['县名']==cy].iloc[0]
            vs = []
            for i in inds:
                v = r[i]
                if pd.isna(v):
                    vs.append(0.5)
                    continue
                yv = df[df['年份']==y][i].dropna()
                mi, ma = yv.min(), yv.max()
                if ma==mi:
                    vs.append(0.5)
                else:
                    if i in ['债务率','土地财政依赖度','财政支出增长率']:
                        vs.append(1-(v-mi)/(ma-mi))
                    else:
                        vs.append((v-mi)/(ma-mi))
            fig.add_trace(go.Scatterpolar(r=vs, theta=inds, fill='toself', name=cy))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0,1])), height=500)
        st.plotly_chart(fig, use_container_width=True)

    # ------------------ 风险地图（st.map 基于真实经纬度，红黄绿三色点） ------------------
    st.markdown("---")
    st.subheader("⏰ 风险地图动态时间轴")
    ys = sorted(df['年份'].unique())
    ty = st.slider("选择年份", min(ys), max(ys), max(ys))
    mdf = df[df['年份']==ty].copy()
    # 颜色映射
    mdf['color'] = mdf['预警等级'].map({'绿灯（健康）':'#00FF00','黄灯（关注）':'#FFCC00','红灯（高风险）':'#FF0000'})
    mdf[['lon','lat']] = mdf['县名'].apply(lambda x: pd.Series(counties_coords.get(x, (106,30))))
    st.map(mdf, latitude='lat', longitude='lon', color='color', zoom=4)

# ------------------ 主程序 ------------------
def main():
    with st.spinner("加载数据..."):
        df, coords = load_data()
    with st.sidebar:
        page = option_menu("导航菜单", ["平台介绍","核心功能","使用指南"], icons=["house","graph-up","book"], menu_icon="cast", default_index=0, styles={"nav-link-selected":{"background-color":"#1e3c72"}})
    if page == "平台介绍":
        platform_intro()
    elif page == "核心功能":
        core_functions(df, coords)
    else:
        usage_guide()

if __name__ == "__main__":
    main()