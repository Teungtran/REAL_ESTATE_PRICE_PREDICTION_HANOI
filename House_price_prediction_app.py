import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
# Load the saved model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
# streamlit run House_price_prediction_app.py
# Define the features
numerical_features = ['floors', 'bedrooms', 'Area(m2)', 'length_meter', 'width_meter']
categorical_features = {
    'housing_type': ['housing_type_Nhà biệt thự', 'housing_type_Nhà mặt phố, mặt tiền', 'housing_type_Nhà ngõ, hẻm', 'housing_type_Nhà phố liền kề'],
    'legal_paper': ['legal_paper_Giấy tờ khác', 'legalpaperĐang chờ sổ', 'legalpaperĐã có sổ'],
    'district': ['district_Ba Đình', 
                'district_Bắc Từ Liê', 
                'district_Cầu Giấy', 
                'district_Hai Bà Trưng', 
                'district_Hoàn Kiế', 
                'district_Hoàng Mai', 
                'district_Huyện Chương Mỹ', 
                'district_Huyện Gia Lâ', 
                'district_Huyện Hoài Đức', 
                'district_Huyện Mê Linh', 
                'district_Huyện Phúc Thọ', 
                'district_Huyện Quốc Oai', 
                'district_Huyện Sóc Sơn', 
                'district_Huyện Thanh Oai', 
                'district_Huyện Thanh Trì', 
                'district_Huyện Thường Tín', 
                'district_Huyện Thạch Thất', 
                'district_Huyện Đan Phượng', 
                'district_Huyện Đông Anh', 
                'district_Hà Đông', 
                'district_Long Biên', 
                'district_Na Từ Liê', 
                'district_Thanh Xuân', 
                'district_Thị xã Sơn Tây', 
                'districtTây Hồ', 
                'districtĐống Đa'],
    'ward': ['ward_Biên Giang','ward_Bách Khoa','ward_Bùi Thị Xuân',
 'ward_Bưởi',
 'ward_Bạch Mai',
 'ward_Bạch Đằng',
 'ward_Bồ Đề',
 'ward_Chương Dương',
 'ward_Cát Linh',
 'ward_Cầu Diễn',
 'ward_Cầu Dền',
 'ward_Cống Vị',
 'ward_Cổ Nhuế 1',
 'ward_Cổ Nhuế 2',
 'ward_Cửa Na',
 'ward_Cửa Đông',
 'ward_Cự Khối',
 'ward_Dương Nội',
 'ward_Dịch Vọng',
 'ward_Dịch Vọng Hậu',
 'ward_Gia Thụy',
 'ward_Giang Biên',
 'ward_Giáp Bát',
 'ward_Giảng Võ',
 'ward_Hoàng Liệt',
 'ward_Hoàng Văn Thụ',
 'ward_Hà Cầu',
 'ward_Hàng Buồ',
 'ward_Hàng Bài',
 'ward_Hàng Bông',
 'ward_Hàng Bồ',
 'ward_Hàng Bột',
 'ward_Hàng Gai',
 'ward_Hàng Mã',
 'ward_Hàng Trống',
 'ward_Hàng Đào',
 'ward_Hạ Đình',
 'ward_Khâ Thiên',
 'ward_Khương Mai',
 'ward_Khương Thượng',
 'ward_Khương Trung',
 'ward_Khương Đình',
 'ward_Ki Giang',
 'ward_Ki Liên',
 'ward_Ki Mã',
 'ward_Kiến Hưng',
 'ward_La Khê',
 'ward_Liên Mạc',
 'ward_Liễu Giai',
 'ward_Long Biên',
 'ward_Láng Hạ',
 'ward_Láng Thượng',
 'ward_Lê Đại Hành',
 'ward_Lý Thái Tổ',
 'ward_Lĩnh Na',
 'ward_Mai Dịch',
 'ward_Mai Động',
 'ward_Minh Khai',
 'ward_Mễ Trì',
 'ward_Mộ Lao',
 'ward_Mỹ Đình 1',
 'ward_Mỹ Đình 2',
 'ward_Na Đồng',
 'ward_Nghĩa Tân',
 'ward_Nghĩa Đô',
 'ward_Nguyễn Du',
 'ward_Nguyễn Trung Trực',
 'ward_Nguyễn Trãi',
 'ward_Ngã Tư Sở',
 'ward_Ngô Quyền',
 'ward_Ngô Thì Nhậ',
 'ward_Ngọc Hà',
 'ward_Ngọc Khánh',
 'ward_Ngọc Lâ',
 'ward_Ngọc Thụy',
 'ward_Nhân Chính',
 'ward_Nhật Tân',
 'ward_Phan Chu Trinh',
 'ward_Phú Diễn',
 'ward_Phú La',
 'ward_Phú Lã',
 'ward_Phú Lương',
 'ward_Phú Thượng',
 'ward_Phú Thịnh',
 'ward_Phú Đô',
 'ward_Phúc Diễn',
 'ward_Phúc La',
 'ward_Phúc Lợi',
 'ward_Phúc Tân',
 'ward_Phúc Xá',
 'ward_Phúc Đồng',
 'ward_Phương Canh',
 'ward_Phương Liên',
 'ward_Phương Liệt',
 'ward_Phương Mai',
 'ward_Phạ Đình Hổ',
 'ward_Phố Huế',
 'ward_Quan Hoa',
 'ward_Quang Trung',
 'ward_Quán Thánh',
 'ward_Quảng An',
 'ward_Quốc Tử Giá',
 'ward_Quỳnh Lôi',
 'ward_Quỳnh Mai',
 'ward_Sài Đồng',
 'ward_Thanh Lương',
 'ward_Thanh Nhàn',
 'ward_Thanh Trì',
 'ward_Thanh Xuân Bắc',
 'ward_Thanh Xuân Na',
 'ward_Thanh Xuân Trung',
 'ward_Thành Công',
 'ward_Thượng Cát',
 'ward_Thượng Thanh',
 'ward_Thượng Đình',
 'ward_Thạch Bàn',
 'ward_Thị trấn Chúc Sơn',
 'ward_Thị trấn Phùng',
 'ward_Thị trấn Quang Minh',
 'ward_Thị trấn Sóc Sơn',
 'ward_Thị trấn Trâu Quỳ',
 'ward_Thị trấn Trạ Trôi',
 'ward_Thị trấn Văn Điển',
 'ward_Thị trấn Yên Viên',
 'ward_Thị trấn Đông Anh',
 'ward_Thịnh Liệt',
 'ward_Thịnh Quang',
 'ward_Thổ Quan',
 'ward_Thụy Khuê',
 'ward_Thụy Phương',
 'ward_Trung Hoà',
 'ward_Trung Liệt',
 'ward_Trung Phụng',
 'ward_Trung Tự',
 'ward_Trung Văn',
 'ward_Tràng Tiền',
 'ward_Trúc Bạch',
 'ward_Trương Định',
 'ward_Trần Hưng Đạo',
 'ward_Trần Phú',
 'ward_Tân Mai',
 'ward_Tây Mỗ',
 'ward_Tây Tựu',
 'ward_Tương Mai',
 'ward_Tứ Liên',
 'ward_Việt Hưng',
 'ward_Văn Chương',
 'ward_Văn Miếu',
 'ward_Văn Quán',
 'ward_Vĩnh Hưng',
 'ward_Vĩnh Phúc',
 'ward_Vĩnh Tuy',
 'ward_Vạn Phúc',
 'ward_Xuân La',
 'ward_Xuân Phương',
 'ward_Xuân Tảo',
 'ward_Xuân Đỉnh',
 'ward_Xã An Khánh',
 'ward_Xã An Thượng',
 'ward_Xã Bích Hòa',
 'ward_Xã Bắc Hồng',
 'ward_Xã Cổ Bi',
 'ward_Xã Cổ Đông',
 'ward_Xã Cự Khê',
 'ward_Xã Di Trạch',
 'ward_Xã Duyên Thái',
 'ward_Xã Dương Liễu',
 'ward_Xã Dương Quang',
 'ward_Xã Hoàng Văn Thụ',
 'ward_Xã Hương Ngải',
 'ward_Xã Hải Bối',
 'ward_Xã Hữu Hoà',
 'ward_Xã Khánh Hà',
 'ward_Xã Ki Chung',
 'ward_Xã Ki Hoa',
 'ward_Xã Ki Sơn',
 'ward_Xã Kiêu Kỵ',
 'ward_Xã La Phù',
 'ward_Xã Liên Ninh',
 'ward_Xã Lê Lợi',
 'ward_Xã Mai Lâ',
 'ward_Xã Na Hồng',
 'ward_Xã Nghĩa Hương',
 'ward_Xã Ngũ Hiệp',
 'ward_Xã Ngọc Hồi',
 'ward_Xã Nhị Khê',
 'ward_Xã Phù Linh',
 'ward_Xã Phù Đổng',
 'ward_Xã Phú Cát',
 'ward_Xã Phú Cường',
 'ward_Xã Phương Trung',
 'ward_Xã Sài Sơn',
 'ward_Xã Sơn Đông',
 'ward_Xã Sơn Đồng',
 'ward_Xã Ta Hiệp',
 'ward_Xã Thanh Liệt',
 'ward_Xã Thượng Mỗ',
 'ward_Xã Thạch Hoà',
 'ward_Xã Tiên Dược',
 'ward_Xã Tiền Phong',
 'ward_Xã Tân Lập',
 'ward_Xã Tân Triều',
 'ward_Xã Tả Thanh Oai',
 'ward_Xã Tứ Hiệp',
 'ward_Xã Vân Canh',
 'ward_Xã Vân Côn',
 'ward_Xã Vân Nội',
 'ward_Xã Võng La',
 'ward_Xã Võng Xuyên',
 'ward_Xã Văn Bình',
 'ward_Xã Vĩnh Quỳnh',
 'ward_Xã Xuân Giang',
 'ward_Xã Yên Thường',
 'ward_Xã Đa Tốn',
 'ward_Xã Đình Xuyên',
 'ward_Xã Đông Dư',
 'ward_Xã Đông Hội',
 'ward_Xã Đông La',
 'ward_Xã Đông Mỹ',
 'ward_Xã Đại Yên',
 'ward_Xã Đại áng',
 'ward_Xã Đặng Xá',
 'ward_Xã Đức Thượng',
 'ward_Yên Hoà',
 'ward_Yên Nghĩa',
 'ward_Yên Phụ',
 'ward_Yên Sở',
 'ward_Yết Kiêu',
 'ward_Ô Chợ Dừa',
 'ward_Điện Biên',
 'ward_Đông Ngạc',
 'ward_Đại Ki',
 'ward_Đại Mỗ',
 'ward_Định Công',
 'ward_Đống Mác',
 'ward_Đồng Mai',
 'ward_Đồng Nhân',
 'ward_Đồng Tâ',
 'ward_Đồng Xuân',
 'ward_Đội Cấn',
 'ward_Đức Giang',
 'ward_Đức Thắng',]
}



# Function to get all possible feature columns
def get_all_feature_columns():
    columns = numerical_features.copy()
    for feature, options in categorical_features.items():
        columns.extend([f"{feature}_{option}" for option in options])
    return columns

# Function for searching options
def search_options(search_term, options):
    return [option for option in options if search_term.lower() in option.lower()]

# Streamlit app
st.title('MÔ HÌNH DỰ ĐOÁN GIÁ NHÀ HÀ NỘI')
st.divider()
st.header('NHẬP THÔNG TIN CHI TIẾT NGÔI NHÀ CỦA BẠN ')

input_data = {}

# Numerical inputs
for feature in numerical_features:
    input_data[feature] = st.number_input(f'Enter {feature}', value=0.0)

# Categorical inputs with search function
for feature, options in categorical_features.items():
    st.subheader(f'Select {feature}')
    search_term = st.text_input(f'Search {feature}', key=f'search_{feature}')
    filtered_options = search_options(search_term, options)
    
    if filtered_options:
        input_data[feature] = st.selectbox(f'Choose {feature}', filtered_options, key=f'select_{feature}')
    else:
        st.warning(f'No matching options found for {feature}. Please try a different search term.')
        input_data[feature] = st.selectbox(f'Choose {feature}', options, key=f'select_{feature}')

# Make prediction button
if st.button('Predict Price'):
    # Convert input_data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Create dummy variables
    input_df_encoded = pd.get_dummies(input_df, columns=list(categorical_features.keys()))
    
    # Ensure all columns from training are present
    all_columns = get_all_feature_columns()
    for col in all_columns:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0
    
    # Reorder columns to match the training data
    input_df_encoded = input_df_encoded[all_columns]
    
    
    # Scale the input data
    input_scaled = scaler.transform(input_df_encoded)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    st.divider()
    
    # Display result
    st.subheader('KẾT QUẢ!')
    st.write(f'Giá/m2 dự kiến  vào khoảng : {prediction[0]:.2f}triệu/m2')

# Add some instructions for the user
st.sidebar.header('HƯỚNG DẪN')
st.sidebar.info('Điền thông tin chi tiết về ngôi nhà vào bảng điều khiển chính và nhấp vào nút "Dự đoán giá" để nhận giá ước tính trên mỗi mét vuông.')
st.sidebar.info('district : Quận/Huyện của ngôi nhà (Ví dụ: Cầu Giấy, Ba Đình, ...)\nward: Phường')

# Add information about the model
st.sidebar.header('THÔNG TIN VỀ MÔ HÌNH')
st.sidebar.info('Mô hình này dự đoán giá nhà ở Hà Nội dựa trên nhiều đặc điểm khác nhau như quy mô, vị trí và tình trạng pháp lý. Nó đã được đào tạo về dữ liệu lịch sử nhà ở trong khu vực.')