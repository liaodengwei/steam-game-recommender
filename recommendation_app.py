import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import requests
from io import BytesIO

# 设置页面标题
st.set_page_config(page_title="Steam 游戏推荐系统", page_icon="🎮", layout="wide")

# 标题和介绍
st.title("🎮 Steam 游戏多模态推荐系统")
st.markdown("""
这个系统基于游戏的**文本描述**和**封面图像**来推荐相似的 Steam 游戏。
选择一款你喜欢的游戏，系统会找到与之最相似的其他游戏。
""")

# 加载数据
@st.cache_data
def load_data():
    # 加载游戏数据
    games_df = pd.read_csv('steam_preprocessed.csv')
    # 加载嵌入向量
    embeddings_df = pd.read_csv('game_embeddings_local.csv')
    
    # 将字符串格式的嵌入向量转换回 numpy 数组
    embeddings_df['text_embedding'] = embeddings_df['text_embedding'].apply(
        lambda x: np.array(eval(x)) if isinstance(x, str) else x
    )
    embeddings_df['image_embedding'] = embeddings_df['image_embedding'].apply(
        lambda x: np.array(eval(x)) if isinstance(x, str) else x
    )
    
    return games_df, embeddings_df

games_df, embeddings_df = load_data()

# 创建游戏选择下拉菜单
game_names = games_df['name'].tolist()
selected_game = st.selectbox("选择一款你喜欢的游戏:", game_names[:100])  # 只显示前100个游戏

# 获取所选游戏的索引
selected_idx = game_names.index(selected_game)

# 权重调节滑块
st.subheader("推荐权重设置")
text_weight = st.slider("文本相似度权重", 0.0, 1.0, 0.5)
image_weight = st.slider("图像相似度权重", 0.0, 1.0, 0.5)

# 确保权重和为1
if text_weight + image_weight != 1.0:
    total = text_weight + image_weight
    text_weight = text_weight / total
    image_weight = image_weight / total
    st.info(f"权重已自动归一化: 文本={text_weight:.2f}, 图像={image_weight:.2f}")

# 计算相似度
if st.button("寻找相似游戏"):
    # 获取所选游戏的嵌入向量
    selected_text_embedding = embeddings_df.iloc[selected_idx]['text_embedding'].reshape(1, -1)
    selected_image_embedding = embeddings_df.iloc[selected_idx]['image_embedding'].reshape(1, -1)
    
    # 准备所有游戏的嵌入向量
    all_text_embeddings = np.vstack(embeddings_df['text_embedding'].values)
    all_image_embeddings = np.vstack(embeddings_df['image_embedding'].values)
    
    # 计算文本相似度
    text_similarities = cosine_similarity(selected_text_embedding, all_text_embeddings)[0]
    
    # 计算图像相似度
    image_similarities = cosine_similarity(selected_image_embedding, all_image_embeddings)[0]
    
    # 计算混合相似度
    hybrid_similarities = text_weight * text_similarities + image_weight * image_similarities
    
    # 排除自身（相似度为1的游戏）
    hybrid_similarities[selected_idx] = 0
    
    # 获取最相似的前5个游戏
    top_5_indices = np.argsort(hybrid_similarities)[-5:][::-1]
    
    # 显示结果
    st.subheader(f"与 '{selected_game}' 最相似的5款游戏:")
    
    # 创建两列布局
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # 显示所选游戏的封面
        try:
            response = requests.get(games_df.iloc[selected_idx]['header_image'])
            img = Image.open(BytesIO(response.content))
            st.image(img, caption=selected_game, use_column_width=True)
        except:
            st.write("无法加载游戏封面")
    
    with col2:
        st.write(f"**游戏名称:** {selected_game}")
        # 显示游戏的类型和标签（如果可用）
        if pd.notna(games_df.iloc[selected_idx]['genres']):
            st.write(f"**类型:** {games_df.iloc[selected_idx]['genres']}")
        if pd.notna(games_df.iloc[selected_idx]['steamspy_tags']):
            st.write(f"**标签:** {games_df.iloc[selected_idx]['steamspy_tags']}")
    
    st.divider()
    
    # 显示推荐的游戏
    for i, idx in enumerate(top_5_indices):
        game_name = games_df.iloc[idx]['name']
        text_sim = text_similarities[idx]
        image_sim = image_similarities[idx]
        hybrid_sim = hybrid_similarities[idx]
        
        # 创建两列布局
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # 显示游戏封面
            try:
                response = requests.get(games_df.iloc[idx]['header_image'])
                img = Image.open(BytesIO(response.content))
                st.image(img, caption=game_name, use_column_width=True)
            except:
                st.write("无法加载游戏封面")
        
        with col2:
            st.write(f"**{i+1}. {game_name}**")
            st.write(f"综合相似度: {hybrid_sim:.4f}")
            st.write(f"文本相似度: {text_sim:.4f}")
            st.write(f"图像相似度: {image_sim:.4f}")
            
            # 显示游戏的类型和标签（如果可用）
            if pd.notna(games_df.iloc[idx]['genres']):
                st.write(f"**类型:** {games_df.iloc[idx]['genres']}")
            if pd.notna(games_df.iloc[idx]['steamspy_tags']):
                st.write(f"**标签:** {games_df.iloc[idx]['steamspy_tags']}")
        
        st.divider()

# 侧边栏信息
st.sidebar.header("关于")
st.sidebar.info("""
这是一个基于多模态（文本+图像）的 Steam 游戏推荐系统演示。

**技术栈:**
- 文本特征: TF-IDF + SVD
- 图像特征: ResNet-50
- 相似度计算: 余弦相似度
- Web框架: Streamlit

数据来源: Steam 游戏数据集 (Kaggle)
""")