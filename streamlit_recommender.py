import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from wordcloud import WordCloud
import re
from underthesea import word_tokenize, pos_tag, sent_tokenize
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import surprise
import math
import difflib


# 1. Read data
ThoiTrangNam_data = pd.read_csv(r'Data_contentbased.csv')
ThoiTrangNam_Raw = pd.read_csv(r'Products_ThoiTrangNam_raw.csv')
Rating_data = pd.read_csv(r'Products_ThoiTrangNam_rating_raw.csv',delimiter='\t')

#--------------
# GUI
st.title("Data Science Project")
st.write("## Project 2 - Shopee Recommender System")
html = '<img src="https://cdn-oss.ginee.com/official/wp-content/uploads/2022/03/image-446-107.png" alt="icon" style="vertical-align: middle; margin-right: 10px; max-width: 100%;">'
st.markdown(f"## {html}", unsafe_allow_html=True)

# GUI
menu = ["Business Objective", "EDA - Exploratory Data Analysis", "Recommender system"]


choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Business Objective':    
    st.subheader("Business Objective")
    st.write("\U0001F3AF **Mục Tiêu/ Vấn đề**")
    st.write("""* Xây dựng Recommendation System cho một hoặc một số nhóm hàng hóa trên shopee.vn giúp đề xuất và gợi ý cho người dùng/ khách hàng.
    """)    
    html = '<img src="https://cdn.vietnambiz.vn/171464876016439296/2020/5/15/rec-15895160961942104685943.jpeg" alt="icon" style="vertical-align: middle; margin-right: 10px; max-width: 100%;">'
    st.markdown(f"## {html}", unsafe_allow_html=True)
    st.write("""
    **Dữ liệu được cung cấp** sẵn gồm có các tập tin:
        ***Products_ThoiTrangNam_raw.csv***,
        ***Products_ThoiTrangNam_rating_raw.csv*** chứa thông tin sản phẩm, review và rating cho các sản phẩm thuộc các nhóm hàng Thời trang nam như Áo khoác, Quần jeans, Áo vest,…
    """)  
    st.write("\U00002753 **Problem/Requirement:**")
    st.write("""* Sử dụng thuật toán Machine Learning trong Python cho Recommender system: **content-based filtering**, **collaborative filtering - user based**""")
    st.image("file_info.png")

    
    html = '<img src="https://www.mdpi.com/electronics/electronics-11-02630/article_deploy/html/images/electronics-11-02630-g001-550.jpg" alt="icon" style="vertical-align: middle; margin-right: 10px; max-width: 100%;">'
    st.markdown(f"## {html}", unsafe_allow_html=True)
    st.subheader("Content-based filtering")
    st.write("""
    Trong Content-based filtering, chúng ta đề xuất các sản phẩm tương tự với các sản phẩm mà người dùng thích (tìm kiếm) dựa trên các thuộc tính của mục đó cho người dùng.

    **Ưu điểm:**
    - Gợi ý được những sản phẩm phù hợp với sở thích của từng khách hàng riêng biệt.
    - Gợi ý không phụ thuộc vào dữ liệu của các khách hàng khác.
    - Gợi ý được những sản phẩm tương tự với những sản phẩm mà khách hàng đã thích trong quá khứ.

    **Hạn chế:**
    - Hồ sơ về sản phẩm nếu không đúng có thể dẫn đến gợi ý sai.
    - Gợi ý phụ thuộc hoàn toàn vào lịch sử của khách hàng. Vì vậy, không thể gợi ý nếu khách hàng không có lịch sử xem/thích các sản phẩm trên hệ thống. Với khách hàng mới, hệ thống không thể cung cấp gợi ý phù hợp.
    - Không gợi ý được các sản phẩm mới, chỉ có thể gợi ý các sản phẩm tương tự như lịch sử đã xem/thích và không gợi ý được các sở thích mới của khách.
    """)  
    st.image("contentbased.png")

    st.subheader("User-Based Collaborative Filtering")
    st.write(""" User-Based Collaborative Filtering: là một kỹ thuật được sử dụng để dự đoán các mặt hàng mà người dùng có thể thích trên cơ sở xếp hạng được đưa ra cho mặt hàng đó bởi những người dùng khác có cùng sở thích với người dùng mục tiêu.
    """)
    st.write(""" Ý tưởng cơ bản của thuật toán này là dự đoán mức độ yêu thích của một user đối với một item dựa trên các users khác “gần giống” với user đang xét. Việc xác định độ “giống nhau” giữa các users có thể dựa vào mức độ quan tâm (rating) của các users này với các items khác mà hệ thống đã biết trong quá khứ.
    """)
    st.write("""
    **Ưu điểm:**

    - Tính đa dạng và cá nhân hóa.

    - Không cần thông tin sản phẩm.

    - Tính linh hoạt.

    **Nhược điểm:**

    - Cold start problem: CF gặp khó khăn khi đối mặt với người dùng mới hoặc sản phẩm mới, gọi là "cold start problem", vì không có đủ thông tin để tạo ra các đề xuất chính xác.

    - Số lượng dữ liệu lớn. 

    - Sparse data: Dữ liệu thường rất thưa.

    - Over-specialization: Mô hình có thể trở nên quá chuyên biệt và chỉ đề xuất các sản phẩm tương tự với nhau, không khuyến khích khám phá sản phẩm mới.

    - Sensitivity to noise and outliers: CF có thể nhạy cảm với nhiễu và dữ liệu ngoại lai, có thể dẫn đến các đề xuất không chính xác.
    """)
    st.image("userbased.png")

    st.subheader("Giáo viên hướng dẫn:")
    st.write("\U0001F469 - Hạ Thị Thiều Dao")
    
    st.subheader("Nhóm Thực hiện:")
    st.write("\U0001F467 - Hạ Thị Thiều Dao")
    st.write("\U0001F466 - Huỳnh Thiện Phúc")
    st.write("\U0001F467 - Văn Thị Tường Vi")

elif choice == 'EDA - Exploratory Data Analysis':
    st.subheader("Exploratory Data Analysis")
    
    st.write('<font color="red">\U0001F4C1 **Products_ThoiTrangNam_raw.csv**</font>', unsafe_allow_html=True)
    st.dataframe(ThoiTrangNam_data.head(5))
    # st.write('Thông tin file:')
    # st.dataframe(ThoiTrangNam_data.describe())

    st.write("\nTổng số sản phẩm:", ThoiTrangNam_data.shape[0])


    st.markdown("#### Average Price by Subcategory")
    # Group by 'sub_category' and calculate the average price
    average_prices = ThoiTrangNam_data.groupby('sub_category')['price'].mean().sort_values(ascending=False)
    # Plotting the average prices using Streamlit components
    st.bar_chart(average_prices)
    # Adding labels and title using markdown text
    st.markdown(f"              **X-axis:** Subcategory")
    st.markdown(f"              **Y-axis:** Average Price")

    st.write("\n \U0001F5EB Sản phẩm có giá cao nhất nằm ở subcategory: Trang phục truyền thống và áo Vest và Blazer.")
    
    # Adding a title above the bar chart using markdown text
    st.markdown("### Average Rating by Subcategory")
    # Plotting the average ratings using Streamlit components
    st.bar_chart(ThoiTrangNam_data.groupby('sub_category')['rating'].mean().sort_values(ascending=False))
    # Adding labels to the axes
    st.markdown("**X-axis:** Subcategory")
    st.markdown("**Y-axis:** Average Rating")

    # Adding a title above the bar chart using markdown text
    st.markdown("### Average Rating by Average Price")
    # Plotting the average ratings using Streamlit components
    st.bar_chart(ThoiTrangNam_data.groupby('rating')['price'].mean())
    # Adding labels to the axes
    st.markdown("**X-axis:** Rating")
    st.markdown("**Y-axis:** Average Price")

    st.write("\n \U0001F5EB Sản phẩm có giá cao thường có rating thấp, rating cao thường có giá trung bình.")

    # Adding a title above the bar chart using markdown text
    st.markdown("### Price vs Rating ")
    # Create a scatter plot using seaborn
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=ThoiTrangNam_data, x='price', y='rating', s=32, alpha=0.8)
    # Remove top and right spines
    plt.gca().spines[['top', 'right']].set_visible(False)
    # Set labels and title
    plt.xlabel('Price')
    plt.ylabel('Rating')
    plt.title('Price vs Rating')
    # Display the plot using Streamlit's pyplot function and pass the figure explicitly
    st.pyplot(plt.gcf())  # plt.gcf() gets the current figure

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Iterate over unique subcategories
    for i, subcategory in enumerate(ThoiTrangNam_data['sub_category'].unique()):
        # Plot histogram of product ratings for each subcategory
        ax.hist(ThoiTrangNam_data[ThoiTrangNam_data['sub_category'] == subcategory]['rating'], alpha=0.5, label=subcategory)

    # Add labels and title
    ax.set_xlabel('Product Rating')
    ax.set_ylabel('Frequency')
    ax.set_title('Product Rating Distribution by Subcategory')
    # Add legend
    ax.legend()
    # Display the plot using Streamlit
    st.pyplot(fig)

    st.dataframe(ThoiTrangNam_data.describe()) 

    # Write the analysis in Streamlit format
    st.write("\U0001F5EB * Giá trung bình của các sản phẩm là khoảng 231,696.5, với độ lệch chuẩn cao, cho thấy sự biến động lớn về giá cả.")
    st.write("\U0001F5EB * Có các sản phẩm có giá từ 0 đến 100,000,000, với hầu hết nằm dưới 270,000.")
    st.write("\U0001F5EB * Đánh giá chủ yếu tập trung giữa 0 và 5, với một số lượng đáng kể các sản phẩm có đánh giá là 0, có thể cho thấy các sản phẩm chưa được đánh giá hoặc đánh giá thấp.")
    st.write("\U0001F5EB * **25% sản phẩm có rating = 0 : sản phẩm không được rate**")

    st.write('<font color="red">\U0001F4C1 **Products_ThoiTrangNam_rating_raw.csv**</font>', unsafe_allow_html=True)
    st.dataframe(Rating_data.head(5))

    # Set seaborn style
    sns.set_style("whitegrid")
    # Plot rating distribution using Seaborn
    plt.figure(figsize=(10, 6))
    sns.histplot(data=Rating_data, x="rating", bins=10, kde=False)
    plt.xlabel("Rating")
    plt.ylabel("Total number of ratings")
    plt.title("Rating Distribution")
    # Display the plot using Streamlit
    st.pyplot(plt)
    
    st.write("#### Products info")
    st.dataframe(Rating_data['product_id'].value_counts().describe()) 

    # Print total data information
    st.write("Total data")
    st.write("-" * 50)
    st.write("\n \U0001F5EB Total number of ratings:", Rating_data.shape[0])
    st.write("\U0001F5EB Total number of users:", len(np.unique(Rating_data.user_id)))
    st.write("\U0001F5EB Total number of products:", len(np.unique(Rating_data.product_id)))

    # Calculate and display the percentage of products with less than 54 ratings
    percent_less_than_54 = (Rating_data[Rating_data['rating'] < 54].shape[0] / Rating_data.shape[0]) * 100
    st.write("\U0001F5EB có 75% product có dưới 54 lượt rating")

elif choice == 'Recommender system':
    tfidf=pickle.load(open('tfidf.pkl','rb'))
    dictionary= pickle.load(open('dictionary.pkl','rb'))
    index= pickle.load(open('index.pkl','rb'))
    BaselineOnly_algorithm= pickle.load(open('BaselineOnly_algorithm.pkl','rb'))
    #algorithm  = pickle.load(open('Model Recommender system_Userbased.sav', 'rb'))
    
    # Function to display image if it's a link
    def display_image(image_url):
        if pd.isna(image_url):
            return None
        elif isinstance(image_url, str) and image_url.startswith("http"):
            return image_url
        else:
            return None
    # Load Vietnamese stopwords
    def load_dict(file_path):
        with open(file_path, 'r', encoding="utf8") as file:
            data = file.read().split('\n')
        dictionary = {}
        for line in data:
            parts = line.split('\t')
            if len(parts) == 2:  # Check if line contains two parts
                dictionary[parts[0]] = parts[1]
        return dictionary

    stopwords_list = load_dict(r'vietnamese-stopwords.txt')

    # Preprocess text using provided functions
    def preprocess_text(text):
        def process_text(text):
            document = text.lower()
            document = document.replace("’",'')
            document = re.sub(r'\.+', ".", document)
            document = document.replace('\n', ' ')
            new_sentence = ''
            for sentence in sent_tokenize(document):
                pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
                sentence = ' '.join(re.findall(pattern, sentence))
                new_sentence += sentence + ' '  # Concatenate sentences
            document = new_sentence.strip()
            document = re.sub(r'\s+', ' ', document).strip()  # Remove excess spaces
            return document

        #def normalize_repeated_characters(text):
        #    return re.sub(r'(.)\1+', r'\1', text)

        def remove_stopword(text, stopwords):
            document = ' '.join('' if word in stopwords else word for word in text.split())
            document = re.sub(r'\s+', ' ', document).strip()
            return document

        def loaddicchar():
            uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
            unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

            dic = {}
            char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
                '|')
            charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
                '|')
            for i in range(len(char1252)):
                dic[char1252[i]] = charutf8[i]
            return dic

        def convert_unicode(txt):
            dicchar = loaddicchar()
            return re.sub(
                r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|AÃ|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
                lambda x: dicchar[x.group()], txt)

        document = process_text(text)
        document = remove_stopword(document, stopwords_list)
        document = convert_unicode(document)
        return document

        # Function to tokenize description input
        def tokenize_description(description_input):
            preprocessed_description = preprocess_text(description_input)
            return preprocessed_description.split()

        # Function to generate word cloud
        def generate_word_cloud(description_list):
            tokens = [word for sublist in description_list for word in sublist]
            word_freq = Counter(tokens)
            most_common_words = word_freq.most_common(30)
            wordcloud_dict = {word: freq for word, freq in most_common_words}
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wordcloud_dict)
            return wordcloud
    # Sidebar for user interaction
    choice = st.sidebar.selectbox("Select functionality", ("1 - Chọn loại sản phẩm", "2 - Tìm kiếm sản phẩm", "3 - Chọn sản phẩm theo ID", "4 - Gợi ý sản phẩm theo User"))

    if choice == "1 - Chọn loại sản phẩm":
        #################################################
        ######  1. Chọn sản phẩm  theo type  ############
        #################################################
        session_state = st.session_state

        #Tạo điều khiển để người dùng chọn sản phẩm
        unique_products = ThoiTrangNam_data['sub_category'].unique().copy()
        # Filter out the category "Khác" (Others)
        unique_products_filtered = [category for category in unique_products if category != 'Khác']

        # Define a custom sorting key function for Vietnamese alphabet
        def vietnamese_sort_key(s):
            # Define the Vietnamese alphabet order
            vietnamese_alphabet = 'aáàảãạăắằẳẵặâấầẩẫậbcdđeéèẻẽẹêếềểễệghiíìỉĩịjklmnoóòỏõọôốồổỗộơớờởỡợpqrstuúùủũụưứừửữựvwxyýỳỷỹỵz'
            # Return the index of each character in the Vietnamese alphabet
            return [vietnamese_alphabet.index(c) if c in vietnamese_alphabet else len(vietnamese_alphabet) for c in s.lower()]
        # Sort the filtered list using the custom sorting key function
        unique_products_sorted_vietnamese = sorted(unique_products_filtered, key=vietnamese_sort_key)   
        st.write("##### 1. Chọn loại sản phẩm")
        selected_SP = st.sidebar.selectbox("Chọn sản phẩm", unique_products_sorted_vietnamese, index= None)
        if selected_SP is not None:
            st.write("Sản phẩm đã chọn:", selected_SP)
            related_SP = ThoiTrangNam_Raw [ThoiTrangNam_Raw['sub_category'].str.lower().str.contains(selected_SP.lower(), na=False)].sort_values(by='rating', ascending=False)
            # In danh sách sản phẩm liên quan ra màn hình
            related_products = related_SP[['product_name', 'image', 'price', 'rating']]

            st.write('<font color="blue">**Danh sách các sản phẩm liên quan:**</font>', unsafe_allow_html=True)

            # for index, row in related_products.iterrows():
            #     st.write(f"**{row['product_name']}**")
            #     image_url = display_image(row['image'])
            #     if image_url:
            #         st.image(image_url, width=200) 
            #     st.write(f"Giá: {row['price']}")
            #     st.write(f"Rating: {row['rating']}")
            #     st.write("---")

            # Define session state to maintain state across interactions
            session_state = st.session_state

            # Assuming each page will display 5 items
            items_per_page = 5

            # Calculate total number of pages
            total_pages = math.ceil(len(related_products) / items_per_page)

            # Get the current page number from session state or default to 1
            current_page = session_state.get('current_page', 1)

            # Handle button clicks
            if st.sidebar.button("Previous Page") and current_page > 1:
                current_page -= 1
            elif st.sidebar.button("Next Page") and current_page < total_pages:
                current_page += 1

            # Update session state
            session_state['current_page'] = current_page

            # Calculate the start and end indices for the current page
            start_index = (current_page - 1) * items_per_page
            end_index = min(start_index + items_per_page, len(related_products))

            # Display pagination controls
            if total_pages > 1:
                st.sidebar.write(f"Page {current_page} of {total_pages}")

            # Display items for the current page
            for index in range(start_index, end_index):
                row = related_products.iloc[index]
                st.write(f"**{row['product_name']}**")
                image_url = display_image(row['image'])
                if image_url:
                    st.image(image_url, width=200) 
                st.write(f"Giá: {row['price']}")
                st.write(f"Rating: {row['rating']}")
                st.write("---")

    elif choice == "2 - Tìm kiếm sản phẩm":
        #################################################
        ######  2. Tìm kiếm Sản phẩm  ###################
        #################################################
        # tạo điều khiển để người dùng tìm kiếm sản phẩm dựa trên thông tin người dùng nhập
        st.write("##### 2. Tìm kiếm sản phẩm") 
        # Define session state to maintain state across interactions
        session_state = st.session_state

        # Your search logic here
        description_input = st.sidebar.text_input("Nhập thông tin tìm kiếm")
        if description_input.strip(): 
            preprocessed_description = preprocess_text(description_input)
            tokenized_description = word_tokenize(preprocessed_description)
            if not tokenized_description:
                st.warning("Thông tin trống. Xin vui lòng nhập nội dung.")
            else:
                query_bow = dictionary.doc2bow(tokenized_description)
                sims = index[tfidf[query_bow]]
                similarity_df = pd.DataFrame({'Document': range(len(sims)), 'Similarity': sims})
                similarity_df['product_name'] = ThoiTrangNam_data['product_name']
                similarity_df = similarity_df.sort_values(by='Similarity', ascending=False)
                similarity_df = similarity_df[similarity_df['Similarity'] < 1]

                # Display top similar products
                st.subheader("Top 10 sản phẩm tương tự:")
                top_10_indices = similarity_df.head(10)['Document'].tolist()
                top_10_descriptions = ThoiTrangNam_data.loc[top_10_indices, 'products_gem_re'].tolist()
                # st.dataframe(ThoiTrangNam_Raw.loc[top_10_indices])
                related_products =ThoiTrangNam_Raw.loc[top_10_indices]

                for index, row in related_products.iterrows():
                    st.write(f"Product Name: {row['product_name']}")
                    image_url = display_image(row['image'])
                    if image_url:
                        st.image(image_url, width=200)
                    st.write(f"Giá: {row['price']}")
                    st.write(f"Rating: {row['rating']}")
                    st.write("---")

                # Generate word cloud for top keywords
                aggregated_description = ' '.join([word for sublist in top_10_descriptions for word in sublist])
                cleaned_description = aggregated_description.replace("[", "").replace("]", ",").replace("'", "").replace(" , ", ", ").replace(" ", "").replace("_", " ").replace(",",", ")
                tokens = cleaned_description.split(",")
                word_freq = Counter(tokens)
                most_common_words = word_freq.most_common(30)
                wordcloud_dict = {word: freq for word, freq in most_common_words}

                # Generate the word cloud
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wordcloud_dict)
                # Plot the word cloud
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title("Word Cloud for Top 30 Most Common Keywords in Similar items")
                # Display the plot using Streamlit
                st.pyplot(fig)

    elif choice == "3 - Chọn sản phẩm theo ID":
        #################################################
        ######  3. Chọn sản phẩm theo ID  ###############
        #################################################
        # Sort the filtered list using the custom sorting key function
            
        # Tạo điều khiển để người dùng chọn sản phẩm
        unique_products_id = ThoiTrangNam_data['product_id'].unique().copy()
        # Filter out the category "Khác" (Others)
        unique_products_id_sorted = sorted(unique_products_id)   
        st.write("##### 3. Chọn sản phẩm theo ID")
        selected_SP3 = st.sidebar.selectbox("Chọn sản phẩm", unique_products_id, index = None)
        if selected_SP3 is not None:
            doc_number = int(selected_SP3)
            # ID sản phẩm luôn nằm trong file
            st.write("Sản phẩm đã chọn:",selected_SP3," - ", ThoiTrangNam_data[ThoiTrangNam_data['product_id'] == doc_number]['product_name'])
            
            selected_product_index = ThoiTrangNam_data[ThoiTrangNam_data['product_id'] == doc_number].index

            # Join the tokenized descriptions into strings
            tokenized_descriptions = [" ".join(tokens) for tokens in ThoiTrangNam_data['products_gem_re']]
            # Apply the desired cleaning operations to each string
            cleaned_descriptions = [desc.replace(" ", "").replace(",", ", ") for desc in tokenized_descriptions]
            # Create TF-IDF vectorizer
            tfidf_vectorizer = TfidfVectorizer()
            # Fit and transform the tokenized descriptions
            tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_descriptions)

            # Compute cosine similarity between the query document and all other documents
            similarities = cosine_similarity(tfidf_matrix[selected_product_index], tfidf_matrix).flatten()

            # Create a DataFrame to store similarity scores and product names
            similarity_df = pd.DataFrame({'Product': range(len(similarities)), 'Similarity': similarities})
            similarity_df['product_name'] = ThoiTrangNam_data['product_name']

            # Sort the DataFrame by similarity scores in descending order
            similarity_df = similarity_df.sort_values(by='Similarity', ascending=False)

            # Exclude the original document itself
            #similarity_df = similarity_df[similarity_df['Product'] != doc_number]

            # Display the top 10 most similar products
            st.write("Top 10 sản phẩm tương tự: ")

            # Get the indices of the top 10 similar products
            top_10_indices = similarity_df.head(10)['Product'].tolist()
            related_products =ThoiTrangNam_Raw.loc[top_10_indices]
            for index, row in related_products.iterrows():
                st.write(f"Product Name: {row['product_name']}")
                image_url = display_image(row['image'])
                if image_url:
                    st.image(image_url, width=200)
                st.write(f"Giá: {row['price']}")
                st.write(f"Rating: {row['rating']}")
                st.write("---")

            # Retrieve the tokenized descriptions of the top 10 similar products from filtered_data
            top_10_descriptions = ThoiTrangNam_data.loc[top_10_indices, 'products_gem_re'].tolist()

            aggregated_description = ' '.join([word for sublist in top_10_descriptions for word in sublist])

            # Remove unnecessary characters and spaces
            cleaned_description = aggregated_description.replace("[", "").replace("]", ",").replace("'", "").replace(" , ", ", ").replace(" ", "").replace("_", " ").replace(",",", ")
            #st.dataframe(cleaned_description)
            # Tokenize the aggregated description
            tokens = cleaned_description.split(",")
            #st.write(tokens)
            # Count the frequency of each word
            word_freq = Counter(tokens)

            # Get the 30 most common words
            most_common_words = word_freq.most_common(30)

            # Create a dictionary of the most common words
            wordcloud_dict = {word: freq for word, freq in most_common_words}

            # Generate the word cloud
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wordcloud_dict)

            # Plot the word cloud
            st.pyplot(plt.figure(figsize=(10, 5)))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title("Word Cloud for Top 30 Most Common Keywords in Similar Items")
            st.pyplot(plt)

    elif choice == "4 - Gợi ý sản phẩm theo User":
        #################################################
        ######  4. Gợi ý sản phẩm theo User ID  #########
        #################################################
        def suggest_similar_user(inputted_user, user_list):
            similar_users = difflib.get_close_matches(inputted_user, user_list, n=5)
            return similar_users
        
        st.write("##### 4. Gợi ý sản phẩm theo User")
        inputted_user = st.sidebar.text_input("Nhập user tìm kiếm")
        if inputted_user.strip():
            # Check if the user ID is valid
            if Rating_data[Rating_data['user'] == inputted_user].shape[0] == 0:
                st.write("User không hợp lệ.")
                # Suggest similar user names
                similar_users = suggest_similar_user(inputted_user, Rating_data['user'])
                if similar_users:
                    st.write("Có thể bạn muốn tìm các user name sau đây:")
                    st.write(similar_users)
                    st.write("Xin vui lòng copy, paste và thử lại.")
            else:
                # Display the top 10 most similar products for the found user
                userid = Rating_data[Rating_data['user'] == inputted_user]['user_id'].iloc[0]

                st.write("Top 10 sản phẩm phù hợp với user ", inputted_user)
                
                df_score = Rating_data[['product_id']]
                df_score['EstimateScore'] = df_score['product_id'].apply(lambda x: BaselineOnly_algorithm.predict(userid, x).est)
                df_score = df_score.sort_values(by=['EstimateScore'], ascending=False)
                df_score = df_score.drop_duplicates()
                merged_data = pd.merge(df_score, ThoiTrangNam_Raw, on='product_id')
                # Display the merged data in a st.dataframe
                related_products =merged_data[['product_name', 'image', 'price', 'EstimateScore', 'rating']].head(10)
                # st.dataframe(merged_data.head(10))
                for index, row in related_products.iterrows():
                    st.write(f"Product Name: {row['product_name']}")
                    image_url = display_image(row['image'])
                    if image_url:
                        st.image(image_url, width=200)
                    st.write(f"Giá: {row['price']}")
                    st.write(f"Rating: {row['rating']}")
                    st.write(f"EstimateRatingScore: {row['EstimateScore']}")
                    st.write("---")
