import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import joblib
import os
import requests
from datetime import datetime


# 1. 数据加载（自动处理编码和列名）
@st.cache_data
def load_data():
    # 尝试多种编码方式
    encodings = ['utf-8', 'gbk', 'utf-16', 'utf-8-sig']

    for encoding in encodings:
        try:
            df = pd.read_csv('crop_data.csv', encoding=encoding)
            # 统一处理列名（去除首尾空格和特殊字符）
            df.columns = df.columns.str.strip().str.replace(' ', '').str.replace('(', '').str.replace(')', '')
            return df
        except (UnicodeDecodeError, FileNotFoundError) as e:
            continue

    st.error("无法加载数据文件，请检查：\n1. 文件是否存在\n2. 文件编码格式")
    st.stop()


# 2. 获取实时天气数据
def get_weather_data(api_key, location):
    """
    从OpenWeatherMap获取实时天气数据
    """
    base_url = "http://api.openweathermap.org/data/2.5/weather"

    try:
        # 尝试通过城市名获取
        params = {
            'q': location,
            'appid': api_key,
            'units': 'metric'  # 获取摄氏温度
        }

        response = requests.get(base_url, params=params)
        data = response.json()

        if response.status_code == 200:
            # 提取所需数据
            weather_data = {
                'temp': data['main']['temp'],
                'rain': data.get('rain', {}).get('1h', 0) * 24 * 30,  # 估算月降雨量(mm)
                'month': datetime.now().month,
                'location': data['name']
            }
            return weather_data
        else:
            st.error(f"天气数据获取失败: {data.get('message', '未知错误')}")
            return None

    except Exception as e:
        st.error(f"获取天气数据时出错: {str(e)}")
        return None


# 3. 模型训练
def train_model(df):
    # 列名映射（处理各种可能的列名变体）
    col_mapping = {
        'month': ['种植月', '月份', 'month'],
        'temp': ['温度℃', '温度', 'temp'],
        'rain': ['降雨mm', '降雨', 'rain'],
        'ph': ['土壤pH', 'pH值', 'ph'],
        'crop': ['作物', 'crop']
    }

    # 自动匹配列名
    selected_cols = {}
    for col_type, possible_names in col_mapping.items():
        for name in possible_names:
            if name in df.columns:
                selected_cols[col_type] = name
                break
        else:
            st.error(f"找不到匹配的列：{possible_names}")
            st.stop()

    X = df[[selected_cols['month'], selected_cols['temp'], selected_cols['rain'], selected_cols['ph']]]
    y = df[selected_cols['crop']]

    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)

    joblib.dump(model, 'model.pkl')
    return model


# 4. 创建Streamlit界面
st.set_page_config(page_title="智能作物推荐系统", layout="wide")
st.title('🌱 智能作物推荐系统')

# 加载数据
try:
    df = load_data()
    st.success("数据加载成功！")
except Exception as e:
    st.error(f"数据加载失败：{str(e)}")
    st.stop()

# 从 secrets.toml 自动加载 API 密钥
try:
    api_key = st.secrets["OPENWEATHERMAP_API_KEY"]  # 这是您要添加的关键行
    st.sidebar.success("API 密钥已自动加载")
except Exception as e:
    st.sidebar.warning("未找到自动加载的API密钥，请手动输入")
    api_key = None

# 在侧边栏添加OpenWeatherMap API配置
with st.sidebar:
    st.header("⚙️ 系统配置")
    if not api_key:  # 如果没有从secrets加载到密钥，则显示输入框
        api_key = st.text_input("OpenWeatherMap API Key", type="password")
    location = st.text_input("地理位置 (城市名)", "Beijing")

    st.header("🛠️ 环境参数")
    use_real_time = st.checkbox("使用实时天气数据", True)

    if not use_real_time:
        month = st.slider("种植月份", 1, 12, 5)
        temp = st.slider("平均温度(℃)", 0, 40, 25)
        rain = st.number_input("降雨量(mm)", 0, 2000, 800)
    else:
        month = datetime.now().month
        temp = 25  # 默认值，会被实时数据覆盖
        rain = 800  # 默认值，会被实时数据覆盖

    ph = st.slider("土壤pH值", 3.0, 9.0, 6.5)

# 主界面
if st.button('🌾 获取推荐'):
    # 如果使用实时数据，获取天气信息
    if use_real_time and api_key:
        with st.spinner('正在获取实时天气数据...'):
            weather_data = get_weather_data(api_key, location)

            if weather_data:
                st.success(f"成功获取 {weather_data['location']} 的天气数据")
                temp = weather_data['temp']
                rain = weather_data['rain']
                month = weather_data['month']

                # 显示天气信息
                st.info(f"""
                **实时天气数据:**
                - 当前温度: {temp:.1f}℃
                - 估算月降雨量: {rain:.1f}mm
                - 当前月份: {month}月
                """)

    try:
        model = joblib.load('model.pkl')
    except FileNotFoundError:
        with st.spinner('首次运行正在训练模型...'):
            model = train_model(df)

    try:
        # 准备输入数据（列顺序必须与训练时一致）
        input_data = pd.DataFrame([[month, temp, rain, ph]],
                                  columns=['种植月', '温度℃', '降雨mm', '土壤pH'])

        # 预测作物
        crop = model.predict(input_data)[0]
        st.success(f'## 推荐种植: {crop}')

        # 显示详细信息
        crop_info = df[df['作物'] == crop].iloc[0]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **📊 种植信息:**
            - 适宜温度: {crop_info['温度℃']}℃
            - 需水量: {crop_info['降雨mm']}mm
            - 最佳pH: {crop_info['土壤pH']}
            """)

        with col2:
            st.markdown(f"""
            **⚠️ 常见问题:**
            - {crop_info['常见问题']}
            - 预计产量: {crop_info['产量等级']}
            """)

    except Exception as e:
        st.error(f"预测失败: {str(e)}")

# 显示原始数据（调试用）
with st.expander("📁 查看原始数据"):
    st.dataframe(df)

# 添加免责声明
st.caption("注意：本推荐结果基于历史数据和实时天气，实际种植请结合当地具体情况。")