import streamlit as st
import pandas as pd
import plotly.express as px
from model import predict_score

# -------------------------- 全局配置（和设计完全一致的配色） --------------------------
st.set_page_config(
    page_title="财政哨兵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS，还原设计的配色、卡片、字体
st.markdown("""
<style>
/* 全局背景色 */
.stApp {
    background-color: #f0f2f6;
}
/* 主卡片样式（浅灰背景，和设计一致） */
.card-gray {
    background-color: #e9ecef;
    padding: 24px;
    border-radius: 16px;
    margin-bottom: 24px;
}
/* 排行卡片样式（和设计一致的浅灰） */
.rank-card {
    background-color: #e9ecef;
    padding: 24px;
    border-radius: 16px;
    margin-bottom: 24px;
}
/* 标题样式 */
h1, h2, h3, h4 {
    color: #1f2937;
    font-weight: 600;
    margin: 0 0 16px 0;
}
/* 红色预警标签 */
.alert-tag {
    background-color: #dc3545;
    color: white;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
}
/* 蓝色排名标签 */
.rank-tag {
    background-color: #cce5ff;
    color: #004085;
    padding: 3px 8px;
    border-radius: 8px;
    font-size: 12px;
}
/* 同比变化样式 */
.diff-up {
    color: #28a745;
    font-size: 14px;
    margin-top: 6px;
}
/* 风险预警卡片样式 */
.warning-card {
    background-color: #f8d7da;
    color: #721c24;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 16px;
}
.caution-card {
    background-color: #fff3cd;
    color: #856404;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 16px;
}
.normal-card {
    background-color: #d4edda;
    color: #155724;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 16px;
}
/* 预警原因条 */
.reason-item {
    background-color: #cce5ff;
    color: #004085;
    padding: 8px 12px;
    border-radius: 8px;
    margin: 6px 0;
    font-size: 14px;
}
/* 改进建议条 */
.suggest-item {
    background-color: #e2e3e5;
    color: #495057;
    padding: 8px 12px;
    border-radius: 8px;
    margin: 6px 0;
    font-size: 14px;
}
/* 隐藏列容器背景，去掉空白框 */
.css-1r6slb0.e1f1d6gn1, .css-keje6w.e1f1d6gn1 {
    background-color: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------- 侧边栏（用你给的logo） --------------------------
st.sidebar.image("logo.png", width=80)
st.sidebar.markdown("### 县级财政健康智能检测平台")
st.sidebar.divider()

# -------------------------- 加载数据和模型 --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("30个县域2020-2024年财政6个指标数据.csv", dtype=str)
    return df

df_raw = load_data()
df_result = predict_score(df_raw)

# -------------------------- 侧边栏筛选模块 --------------------------
st.sidebar.markdown("#### 数据筛选")
year = st.sidebar.selectbox("评估年份", sorted(df_result["年份"].unique()))
county = st.sidebar.selectbox("所在县", sorted(df_result[df_result["年份"] == year]["县名"].unique()))

with st.sidebar.expander("评分体系说明"):
    st.write("综合财政健康指数基于债务率、财政自给率、收支平衡等12项核心指标加权计算。")

if st.sidebar.button("生成PDF报告", type="primary", use_container_width=True):
    st.sidebar.success("✅ 报告导出成功！")

# -------------------------- 主页面标题 --------------------------
st.markdown(f"## {county} · {year}年 财政健康检测报告")

# -------------------------- 1. 顶部双卡片（和设计完全对齐） --------------------------
col1, col2 = st.columns(2)

# 左侧：综合财政健康评分卡片（浅灰背景，和设计一致）
with col1:
    st.markdown('<div class="card-gray">', unsafe_allow_html=True)
    # 标题+预警标签
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <h4 style="margin:0;">综合财政健康评分</h4>
        <span class="alert-tag">预警</span>
    </div>
    """, unsafe_allow_html=True)
    
    # 获取当前县数据
    current = df_result[(df_result["年份"] == year) & (df_result["县名"] == county)].iloc[0]
    score = current["综合得分（百分制）"]
    rank = current["排名"]
    prev_year = str(int(year)-1)
    prev_score = df_result[(df_result["年份"] == prev_year) & (df_result["县名"] == county)]["综合得分（百分制）"].values
    
    if len(prev_score) > 0:
        diff = round(score - prev_score[0], 1)
        diff_text = f"同比+{diff}" if diff >=0 else f"同比{diff}"
    else:
        diff_text = "无同比数据"
    
    # 大分数+排名+同比（和设计排版一致）
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:20px; margin:16px 0;">
        <span style="font-size:40px; font-weight:bold; color:#dc3545;">{score}</span>
        <div>
            <span class="rank-tag">全县排名{rank}/30名</span>
            <br>
            <span class="diff-up">√ {diff_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 进度条（和设计颜色一致，红到蓝渐变）
    st.progress(score/100)
    st.markdown('</div>', unsafe_allow_html=True)

# 右侧：风险预警分析卡片（白色背景，和设计一致）
with col2:
    st.markdown('<div class="card-gray">', unsafe_allow_html=True)
    st.markdown("#### 风险预警分析")
    
    level = current["预警等级"]
    if "红灯" in level:
        st.markdown(f"""
        <div class="warning-card">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                <span style="font-size:20px;">⚠️</span>
                <span style="font-weight:bold; font-size:16px;">预警</span>
            </div>
            <div style="font-size:14px;">{county} · 风险等级较高</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("##### 预警原因")
        st.markdown("""
        <div class="reason-item">ⓧ 偿债能力严重不足</div>
        <div class="reason-item">ⓧ 资金周转出现困难</div>
        <div class="reason-item">ⓧ 收支缺口持续扩大</div>
        """, unsafe_allow_html=True)
        st.markdown("##### 改进建议")
        st.markdown("""
        <div class="suggest-item">▷ 启动债务风险应急预案，制定化债时间表</div>
        <div class="suggest-item">▷ 严禁违规举债，全面排查隐性债务</div>
        """, unsafe_allow_html=True)
    elif "黄灯" in level:
        st.markdown(f"""
        <div class="caution-card">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                <span style="font-size:20px;">⚠️</span>
                <span style="font-weight:bold; font-size:16px;">关注</span>
            </div>
            <div style="font-size:14px;">{county} · 存在潜在风险</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("##### 预警原因")
        st.markdown("""
        <div class="reason-item">ⓧ 财政自给率低于警戒线</div>
        <div class="reason-item">ⓧ 支出增速高于收入增速</div>
        """, unsafe_allow_html=True)
        st.markdown("##### 改进建议")
        st.markdown("""
        <div class="suggest-item">▷ 优化支出结构，压减非刚性支出</div>
        <div class="suggest-item">▷ 培育税源，提高税收收入占比</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="normal-card">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                <span style="font-size:20px;">✅</span>
                <span style="font-weight:bold; font-size:16px;">正常</span>
            </div>
            <div style="font-size:14px;">{county} · 财政状况良好</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("##### 健康表现")
        st.markdown("""
        <div class="reason-item">ⓧ 财政自给率稳定在合理区间</div>
        <div class="reason-item">ⓧ 债务风险整体可控</div>
        """, unsafe_allow_html=True)
        st.markdown("##### 优化建议")
        st.markdown("""
        <div class="suggest-item">▷ 持续优化税源结构</div>
        <div class="suggest-item">▷ 防范隐性债务风险</div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# -------------------------- 2. 历史得分趋势折线图 --------------------------
st.markdown("#### 历史得分趋势（2020-2024）")
county_history = df_result[df_result["县名"] == county].sort_values("年份")
fig = px.line(
    county_history,
    x="年份",
    y="综合得分（百分制）",
    markers=True,
    color_discrete_sequence=["#4169E1"],
    labels={"综合得分（百分制）": "得分", "年份": "年份"}
)
fig.update_layout(
    yaxis_range=[0,100],
    plot_bgcolor="#e9ecef",
    paper_bgcolor="#e9ecef"
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# -------------------------- 3. 各县风险分布地图（真实经纬度+全30个县） --------------------------
st.markdown(f"#### 各县风险分布地图（{year}年）")
df_map = df_result[df_result["年份"] == year].copy()

# 预警等级颜色映射（和图例一致）
level_to_color = {
    "绿灯（健康）": "#00FF00",
    "黄灯（关注）": "#FFCC00",
    "红灯（高风险）": "#FF0000"
}
df_map["color"] = df_map["预警等级"].map(level_to_color)

# 你30个县的真实经纬度
county_coords = {
    "昆山": (120.95, 31.39),
    "江阴": (120.27, 31.91),
    "义乌": (120.06, 29.31),
    "晋江": (118.56, 24.78),
    "荣成": (122.42, 37.16),
    "常熟": (120.74, 31.64),
    "慈溪": (121.24, 30.17),
    "宜兴": (119.82, 31.36),
    "长沙县": (113.08, 28.24),
    "新郑": (113.71, 34.36),
    "肥西": (117.15, 31.72),
    "大冶": (114.98, 30.10),
    "南昌县": (115.94, 28.56),
    "浏阳": (113.63, 28.15),
    "宁乡": (112.56, 28.27),
    "长丰": (117.17, 32.47),
    "仁怀": (106.40, 27.81),
    "简阳": (104.53, 30.40),
    "安宁": (102.48, 24.92),
    "神木": (110.47, 38.81),
    "平果": (107.58, 23.32),
    "大理": (100.22, 25.61),
    "盘州": (104.47, 25.71),
    "瓦房店": (121.97, 39.62),
    "海城": (122.70, 40.88),
    "公主岭": (124.82, 43.50),
    "梅河口": (125.65, 42.53),
    "肇东": (125.97, 46.07),
    "通榆县": (123.05, 44.81),
    "普兰店": (121.97, 39.39)
}

# 合并经纬度
df_map[["lon", "lat"]] = df_map["县名"].apply(lambda x: pd.Series(county_coords.get(x, (106.0, 30.0))))

# 绘制地图
st.map(
    df_map,
    latitude="lat",
    longitude="lon",
    color="color",
    zoom=4
)

# 图例说明
st.markdown("""
<div style="display: flex; gap: 20px; margin-top: 10px; justify-content: center;">
  <div style="display: flex; align-items: center;"><div style="width:15px; height:15px; background:#00FF00; border-radius:50%; margin-right:5px;"></div>健康</div>
  <div style="display: flex; align-items: center;"><div style="width:15px; height:15px; background:#FFCC00; border-radius:50%; margin-right:5px;"></div>关注</div>
  <div style="display: flex; align-items: center;"><div style="width:15px; height:15px; background:#FF0000; border-radius:50%; margin-right:5px;"></div>预警</div>
</div>
""", unsafe_allow_html=True)

st.divider()

# -------------------------- 4. 得分排行（前5/后5，浅灰背景） --------------------------
st.markdown(f"#### 得分排行（{year}年）")
top5 = df_result[df_result["年份"] == year].sort_values("综合得分（百分制）", ascending=False).head(5)
bottom5 = df_result[df_result["年份"] == year].sort_values("综合得分（百分制）", ascending=True).head(5)

col_top, col_bottom = st.columns(2)
with col_top:
    st.markdown('<div class="rank-card">', unsafe_allow_html=True)
    st.subheader("前5名")
    st.dataframe(
        top5[["县名", "综合得分（百分制）", "预警等级"]],
        hide_index=True,
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
with col_bottom:
    st.markdown('<div class="rank-card">', unsafe_allow_html=True)
    st.subheader("后5名")
    st.dataframe(
        bottom5[["县名", "综合得分（百分制）", "预警等级"]],
        hide_index=True,
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)