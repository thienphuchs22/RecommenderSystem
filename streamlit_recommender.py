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

# 1. Read data
ThoiTrangNam_data = pd.read_csv(r'Data_contentbased.csv')
Rating_data = pd.read_csv(r'Products_ThoiTrangNam_rating_raw.csv',delimiter='\t')

#--------------
# GUI
st.title("Data Science Project")
st.write("## Project 2 - Shopee Recommender System")

# GUI
menu = ["Business Objective", "EDA - Exploratory Data Analysis", "Recommender system"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Business Objective':    
    st.subheader("Business Objective")
    st.write("""
    **Mục tiêu/ vấn đề** : 
    """)
    st.write("""* Xây dựng Recommendation System cho một hoặc một số nhóm hàng hóa trên shopee.vn giúp đề xuất và gợi ý cho người dùng/ khách hàng.
    """)
    st.write("""
    **Dữ liệu được cung cấp** sẵn gồm có các tập tin:
        ***Products_ThoiTrangNam_raw.csv***,
        ***Products_ThoiTrangNam_rating_raw.csv*** chứa thông tin sản phẩm, review và rating cho các sản phẩm thuộc các nhóm hàng Thời trang nam như Áo khoác, Quần jeans, Áo vest,…
    """)  
    st.write("""###### => Problem/ Requirement:""")
    st.write("""* Use Machine Learning algorithms in Python for system recommender: content- based filtering, collaborative filtering - user based""")
    st.image("file_info.png")


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
    st.image("userbased.png")
    st.subheader("Công nghệ sử dụng")
    st.write(""" 
    - gensim
    - cosine similarity
    - surprise 
    """)
    st.subheader("Author:")
    st.write(""" 
    - Hạ Thị Thiều Dao
    - Huỳnh Thiện Phúc
    - Văn Thị Tường Vi
    """)

elif choice == 'EDA - Exploratory Data Analysis':
    st.subheader("Exploratory Data Analysis")
    
    st.write('<font color="red">**Products_ThoiTrangNam_raw.csv**</font>', unsafe_allow_html=True)
    st.dataframe(ThoiTrangNam_data.head(3))


    st.markdown("#### Average Price by Subcategory")
    # Group by 'sub_category' and calculate the average price
    average_prices = ThoiTrangNam_data.groupby('sub_category')['price'].mean().sort_values(ascending=False)
    # Plotting the average prices using Streamlit components
    st.bar_chart(average_prices)
    # Adding labels and title using markdown text
    st.markdown(f"              **X-axis:** Subcategory")
    st.markdown(f"              **Y-axis:** Average Price")
    
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
    st.write("* Giá trung bình của các sản phẩm là khoảng 231,696.5, với độ lệch chuẩn cao, cho thấy sự biến động lớn về giá cả.")
    st.write("* Có các sản phẩm có giá từ 0 đến 100,000,000, với hầu hết nằm dưới 270,000.")
    st.write("* Đánh giá chủ yếu tập trung giữa 0 và 5, với một số lượng đáng kể các sản phẩm có đánh giá là 0, có thể cho thấy các sản phẩm chưa được đánh giá hoặc đánh giá thấp.")
    st.write("* **25% sản phẩm có rating = 0 : sản phẩm không được rate**")

    st.write('<font color="red">**Products_ThoiTrangNam_rating_raw.csv**</font>', unsafe_allow_html=True)
    st.dataframe(Rating_data.head(3))

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
    st.write("\nTotal number of ratings:", Rating_data.shape[0])
    st.write("Total number of users:", len(np.unique(Rating_data.user_id)))
    st.write("Total number of products:", len(np.unique(Rating_data.product_id)))

    # Calculate and display the percentage of products with less than 54 ratings
    percent_less_than_54 = (Rating_data[Rating_data['rating'] < 54].shape[0] / Rating_data.shape[0]) * 100
    st.write("có 75% product có dưới 54 lượt rating")

elif choice == 'Recommender system':
    tfidf=pickle.load(open('tfidf.pkl','rb'))
    dictionary= pickle.load(open('dictionary.pkl','rb'))
    index= pickle.load(open('index.pkl','rb'))
    #algorithm  = pickle.load(open('Model Recommender system_Userbased.sav', 'rb'))
    
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
    choice = st.sidebar.selectbox("Select functionality", ("1 - Chọn loại sản phẩm", "2 - Tìm kiếm sản phẩm", "3 - Chọn sản phẩm theo ID", "4 - Gợi ý sản phẩm theo User ID"))

    if choice == "1 - Chọn loại sản phẩm":
        #################################################
        ######  1. Chọn sản phẩm  theo type  ############
        #################################################
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
        selected_SP = st.sidebar.selectbox("Chọn sản phẩm", unique_products_sorted_vietnamese )
        st.write("Sản phẩm đã chọn:", selected_SP)

        # Tìm sản phẩm liên quan đến sản phẩm đã chọn
        # st.write("##### 2. Sản phẩm liên quan")
        # Lấy thông tin sản phẩm đã chọn
        # Gợi ý sản phẩm liên quan dựa theo mô tả của sản phẩm đã chọn, chuyển thành chữ thường trước khi tìm kiếm 
        related_SP = ThoiTrangNam_data[ThoiTrangNam_data['sub_category'].str.lower().str.contains(selected_SP.lower(), na=False)].sort_values(by='rating', ascending=False)
        # In danh sách sản phẩm liên quan ra màn hình
        st.write("Danh sách sản phẩm liên quan:")
        st.dataframe(related_SP.head(5))   
        # Từ sản phẩm đã chọn này, người dùng có thể xem thông tin chi tiết của sản phẩm, xem hình ảnh sản phẩm
        # hoặc thực hiện các xử lý khác

    elif choice == "2 - Tìm kiếm sản phẩm":
        #################################################
        ######  2. Tìm kiếm Sản phẩm  ###################
        #################################################
        # tạo điều khiển để người dùng tìm kiếm sản phẩm dựa trên thông tin người dùng nhập
        st.write("##### 2. Tìm kiếm sản phẩm") 
        description_input = st.sidebar.text_input("Nhập thông tin tìm kiếm")
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

            st.subheader("Top 10 sản phẩm tương tự:")

            top_10_indices = similarity_df.head(10)['Document'].tolist()
            top_10_descriptions = ThoiTrangNam_data.loc[top_10_indices, 'products_gem_re'].tolist()
            st.dataframe(ThoiTrangNam_data.iloc[top_10_indices])

            # Concatenate tokens and remove unwanted characters
            aggregated_description = ' '.join([word for sublist in top_10_descriptions for word in sublist])
            # Remove unnecessary characters and spaces
            cleaned_description = aggregated_description.replace("[", "").replace("]", ",").replace("'", "").replace(" , ", ", ").replace(" ", "").replace("_", " ").replace(",",", ")
            # Display aggregated_description
            #st.write(cleaned_description)
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
        selected_SP3 = st.sidebar.selectbox("Chọn sản phẩm", unique_products_id)
        doc_number = int(selected_SP3)
        st.write("Sản phẩm đã chọn:",selected_SP3," - ", ThoiTrangNam_data['product_name'].iloc[doc_number])
        
        # Join the tokenized descriptions into strings
        tokenized_descriptions = [" ".join(tokens) for tokens in ThoiTrangNam_data['products_gem_re']]
        # Apply the desired cleaning operations to each string
        cleaned_descriptions = [desc.replace(" ", "").replace(",", ", ") for desc in tokenized_descriptions]
        # Create TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer()
        # Fit and transform the tokenized descriptions
        tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_descriptions)
        
        
        # Check if the document number is valid
        if doc_number < 0 or doc_number >= len(ThoiTrangNam_data):
            st.write("Không có sản phẩm với ID ",selected_SP3)
        else:

            # Compute cosine similarity between the query document and all other documents
            similarities = cosine_similarity(tfidf_matrix[doc_number], tfidf_matrix).flatten()

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
            st.dataframe(ThoiTrangNam_data.iloc[top_10_indices])

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

    elif choice == "4 - Gợi ý sản phẩm theo User ID":
        #################################################
        ######  4. Gợi ý sản phẩm theo User ID  #########
        #################################################
        # Sort the filtered list using the custom sorting key function
        
        #st.dataframe(Rating_data.head())
        # Tạo điều khiển để người dùng chọn sản phẩm
        unique_user_id = Rating_data['user_id'].unique().copy()
        # Filter out the category "Khác" (Others)
        unique_user_id_sorted = sorted(unique_user_id)   
        #st.dataframe(unique_user_id_sorted)
        
        st.write("##### 4. Gợi ý sản phẩm theo User ID")
        
        inputted_user = st.sidebar.text_input("Nhập user id tìm kiếm")
 
       
        
        # Check if the document number is valid
        if inputted_user not in unique_user_id and not inputted_user.isdigit():
            st.write("User ID không hợp lệ. Vui lòng nhập một user ID khác.")
        else:
            # Display the top 10 most similar products
            userid =int(inputted_user)
            st.write("Top 10 sản phẩm user ",inputted_user," đã mua:")
            df_select = Rating_data[(Rating_data['user_id'] == userid) & (Rating_data['rating'] >=3)]
            df_select = df_select.sort_values(by='rating', ascending=False)

            # Merge Rating_data with ThoitrangNam_data based on product_id
            merged_data = pd.merge(df_select, ThoiTrangNam_data, on='product_id', how='inner')

            # Display the merged data in a st.dataframe
            st.dataframe(merged_data.head(10))