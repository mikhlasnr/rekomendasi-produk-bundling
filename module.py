import streamlit as st
from streamlit import session_state
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, fpgrowth
from io import BytesIO
# import os

class AssociationRules:
    def MemeriksaUploadTransaksi():
        # Initialize the session_state dictionary if it doesn't exist
        if 'upload_transactions' not in st.session_state:
            session_state.upload_transactions = None
        if 'df_transactions' not in st.session_state:
            session_state.df_transactions = pd.DataFrame()
            
        # Memeriksa apakah 'upload_transactions' tidak kosong dan DataFrame 'df_transactions' tidak kosong
        return session_state.get('upload_transactions') is not None and not session_state.df_transactions.empty
     
    def setUploadTransaksi(newUploadTransaction, newDFTransactions):
        session_state.upload_transactions = newUploadTransaction
        session_state.df_transactions = newDFTransactions

    def getUploadTransaksi():
        if AssociationRules.MemeriksaUploadTransaksi():
            return session_state.df_transactions
        return pd.DataFrame()
    
    def memeriksaListProduk():
        return session_state.get('list_produk') is not None
    
    def setListProduk(listProduk):
        session_state.list_produk = listProduk

    def getListProduk():
        if AssociationRules.memeriksaListProduk():
            return session_state.list_produk
        return pd.DataFrame()
    
    def memeriksaRules():
        rules_df = session_state.get('rules')
        return rules_df is not None and not rules_df.empty
    
    def memasukkanTransaksi():
        if AssociationRules.getUploadTransaksi().empty:
            # File uploader
            st.markdown("<h4>Upload data transaksi penjualan untuk mendapatkan rekomendasi produk bundling dan membuat paket bundling</h4>", unsafe_allow_html=True)
            upload = st.file_uploader("Choose a CSV file", type="csv")
            return upload
        else:
            st.success("Berhasil memasukkan file transaksi")
            if st.button("Ganti data transaksi"):
                AssociationRules.setUploadTransaksi(None, None)
                st.experimental_rerun()

    def validasiMasukkanTransaksi(file):
        if file:
            dataframe_transaction = pd.read_csv(file) 
            # Mengecek apakah kolom yang dibutuhkan ada di DataFrame
            required_columns = ['Name', 'Lineitem name', 'Lineitem price']
            missing_columns = [col for col in required_columns if col not in dataframe_transaction.columns]
            if len(missing_columns) == 0:
                AssociationRules.setUploadTransaksi(file, dataframe_transaction)
                st.experimental_rerun()
            else:
                st.error("File transaksi harus merupakan transaksi Greenspace.id")
        

    @st.cache_data
    def selectAttribute(transactions):
        # Ambil atribut tertentu
        select_attribute = transactions.copy()
        select_attribute = select_attribute[['Name', 'Lineitem name', 'Lineitem price']]
        return select_attribute
    
    def transformText(select_attribute):
        # Ambil atribut tertentu
        select_attribute_uppercase = select_attribute.copy()
        select_attribute_uppercase["Lineitem name"] = select_attribute_uppercase["Lineitem name"].str.upper()
        return select_attribute_uppercase

    @st.cache_data
    def dataCleaning(transformText):
        filter_outlier = transformText.copy()
        # Menghapus outlier dengan menghilangkan data yang mengandung 'Pcs', 'Certificate', dan 'Bundling'
        filter_outlier = filter_outlier[~filter_outlier['Lineitem name'].str.contains(r'\[FLASH SALE\]|\[FLASHSALE\]|PCS|CERTIFICATE|BUNDLING')]
        return filter_outlier

    def transformData(dataCleaning):
        # Get all the transactions as a list of lists
        transactions_list = [transaction[1]['Lineitem name'].tolist() for transaction in list(dataCleaning.groupby(['Name']))]
        
        # The following instructions transform the dataset into the required format 
        trans_encoder = TransactionEncoder()
        trans_encoder_matrix = trans_encoder.fit(transactions_list).transform(transactions_list)
        trans_encoder_matrix = pd.DataFrame(trans_encoder_matrix, columns=trans_encoder.columns_)
        
        return transactions_list, trans_encoder_matrix
    
    def createListProduk(dataCleaning):
        if 'list_produk' not in session_state:
            session_state.list_produk = None
    
        list_produk = dataCleaning.copy()
        # Membuat tabel baru 'list_produk' dengan kolom 'Nama tanaman' dan 'Lineitem price'
        list_produk = list_produk[['Lineitem name', 'Lineitem price']]
        # Menghapus data duplikat pada nilai 'Lineitem name' dan memilih 'Lineitem price' yang paling besar
        list_produk = list_produk.drop_duplicates(subset=['Lineitem name'])
        # Harga Beli
        list_produk['Harga Beli'] = (list_produk['Lineitem price'] / 4).round(2)
        
        # Harga Minimum Jual
        list_produk['Harga Minimum Jual'] = (list_produk['Harga Beli'] * 3).round(2)
        
        # Menghitung kemunculan setiap nilai pada Lineitem name dari tabel produk pada transaksi
        list_produk['Support'] = list_produk['Lineitem name'].map(dataCleaning['Lineitem name'].value_counts()).fillna(0).astype(int)
        
        # Mengurutkan Support dari terbesar ke terkecil
        list_produk = list_produk.sort_values(by='Lineitem name', ascending=True)
        list_produk.reset_index(drop=True, inplace=True)

        AssociationRules.setListProduk(list_produk)
        return list_produk
    
    def minSupport(list_produk, transactions_list):
        get_frekuensi = list_produk['Support'].mean()
        get_frekuensi = round(get_frekuensi)
        get_len_transaction = len(transactions_list)
        # Menghitung persentase
        min_support = get_frekuensi / get_len_transaction
        min_support = round(get_frekuensi / len(transactions_list), 2)
        return min_support

    def rules(trans_encoder_matrix, min_support):
        min_confidence = 0.7
        if 'rules' not in session_state:
            session_state.rules = None

        if 'frequent_itemsets' not in session_state:
            session_state.frequent_itemsets = pd.DataFrame()

        if 'df_association' not in session_state:
            session_state.df_association = pd.DataFrame()

        if 'df_association_unique' not in session_state:
            session_state.df_association_unique = pd.DataFrame()
        
        # Membuat frequent itemsets
        frequent_itemsets = fpgrowth(
            trans_encoder_matrix, min_support=min_support, use_colnames=True)
        session_state.frequent_itemsets = frequent_itemsets
        
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        rules['Count items'] = rules['antecedents'].apply(lambda x: len(x)) + rules['consequents'].apply(lambda x: len(x))
        rules['id_rule'] = rules.reset_index().index + 1
        session_state.rules = rules

        # Membuat dataframe dari data association rule
        df_association = rules.copy()
        # Combine antecedents and consequents into a new column 'produk rules' using list comprehension
        df_association['Produk Rules'] = [sorted(list(row['antecedents']) + list(row['consequents'])) for _, row in df_association.iterrows()]
        df_association.drop(columns=['id_rule','antecedents', 'consequents', 'antecedent support', 'consequent support', 'support', 'confidence', 'lift', 'leverage','conviction', 'zhangs_metric', 'Count items'], inplace=True)
        session_state.df_association = df_association

        # Mengkonversi daftar menjadi string
        df_association_unique = df_association.copy()
        df_association_unique['Produk Rules'] = df_association_unique['Produk Rules'].astype(str)

        # Menghilangkan duplikat berdasarkan nilai pada kolom 'Produk Rules'
        df_association_unique = df_association_unique.drop_duplicates(subset='Produk Rules')

        # Mengembalikan nilai daftar dari string (opsional, tergantung pada kebutuhan)
        df_association_unique['Produk Rules'] = df_association_unique['Produk Rules'].apply(eval)
        df_association_unique = df_association_unique.reset_index(drop=True)
        session_state.df_association_unique = df_association_unique

        return rules, frequent_itemsets
    
    @st.cache_data
    def tampilProsesAssociationRules():
        # MENAMPILKAN DATA TRANSAKSI
        df_transactions = AssociationRules.getUploadTransaksi()
        st.markdown("<h2>Data Transaksi</h2>",unsafe_allow_html=True)
        st.write("Terdapat: ", df_transactions.shape[0], "record dan", df_transactions.shape[1], "attribute")
        st.dataframe(df_transactions)
        st.divider()

        # MENAMPILKAN PREPARATION
        st.markdown("<h2>Preparation</h2>",unsafe_allow_html=True)
        
        # MENAMPILKAN PREPARATION SELECT ATTRIBUTE
        st.markdown(
            "<h3>Select attribute</h3>", unsafe_allow_html=True)
        st.markdown("<p>Memilih attribute yang dibutuhkan yaitu Name dan Lineitem name</p>",unsafe_allow_html=True)
        selectAttribute = AssociationRules.selectAttribute(df_transactions)
        st.dataframe(selectAttribute)

        # MENAMPILKAN PREPARATION TRANSFORM TEXT
        st.markdown("<h3>Uppercase Lineitem name</h3>", unsafe_allow_html=True)
        st.markdown("<p>Karena penamaan tanaman terdapat yang tidak konsiten dalam huruf besar kecilnya maka perlu dilakukan transformasi nilai pada attribute tersebut menjadi huruf capital</p>",unsafe_allow_html=True)
        transformText = AssociationRules.transformText(selectAttribute)
        st.dataframe(transformText)

        # MENAMPILKAN PREPARATION DATA CLEANING
        st.markdown("<h3>DATA CLEANING</h3>", unsafe_allow_html=True)
        st.markdown("<h4>Memindai Data Null</h4>", unsafe_allow_html=True)
        st.write(transformText.isnull().sum())
        st.markdown("<p>Diketahui tidak terdapat nilai null pada nilai attribute data yang akan digunakan</p>", unsafe_allow_html=True)
        st.markdown("<h3>Menghapus outlier pada nilai attribute Lineitem name yang mengandung kalimat kata PCS, CERTIFICATE dan BUNDLING</h3>", unsafe_allow_html=True)
        st.markdown("<p>Berikut adalah data yang memiliki nilai outlier</p>", unsafe_allow_html=True)
        filter_outlier = transformText.copy()
        # Menampilkan data pada kolom 'Lineitem name' yang mengandung 'Pcs', 'Certificate', atau 'Bundling'
        transactions_filter_outlier_show = filter_outlier[filter_outlier['Lineitem name'].str.contains(r'\[FLASH SALE\]|\[FLASHSALE\]|PCS|CERTIFICATE|BUNDLING')]
        st.write("terdapat", transactions_filter_outlier_show.shape[0], 'record yang mengandung outlier')
        st.dataframe(transactions_filter_outlier_show)
        # Menampilkan data hasil cleaning data
        dataCleaning = AssociationRules.dataCleaning(transformText)
        st.markdown("<p>Berikut adalah data setelah dilakukan dibersihkan</p>", unsafe_allow_html=True)
        st.write("sehingga data yang digunakan saat ini memiliki ", dataCleaning.shape[0], "records")
        st.dataframe(dataCleaning)
        
        # MENAMPILKAN PREPARATION TRANSFORMASI DATA
        st.markdown(
            "<h3 >Transformasi Data</h3>", unsafe_allow_html=True)
        st.markdown(
            "<h4>Mengabungkan nilai Lineitem name menjadi satu list berdasarkan nilai Name</h4>", unsafe_allow_html=True)
        transactions_list, trans_encoder_matrix = AssociationRules.transformData(dataCleaning)
        st.write("Diketahui terdapat", len(transactions_list), 'transaksi')
        st.write(transactions_list)
        st.markdown(
            "<h4>Transformasi data kebentuk yang dibutuhkan untuk modelling</h4>", unsafe_allow_html=True)
        st.dataframe(trans_encoder_matrix)
        st.divider()
        
        # MENAMPILKAN MODLING
        # MENAMPILKAN MODELING MIN SUPPORT
        st.markdown("<h2>Modeling</h2>",unsafe_allow_html=True)
        st.markdown("<h3>Minimum Support</h3>", unsafe_allow_html=True)
        st.markdown("<p>min support didapatkan dari rata-rata Support produk sehingga perlu dibuat list produk dengan Support. Lineitem price akan digunakan untuk membuat rekomendasi paket bundling</p>", unsafe_allow_html=True)
        list_produk = AssociationRules.getListProduk()
        list_produk_show = list_produk[['Lineitem name','Support']]
        st.dataframe(list_produk_show, use_container_width=True)
        get_frekuensi = list_produk['Support'].mean()
        get_frekuensi = round(get_frekuensi)
        get_len_transaction = len(transactions_list)
        st.write("Diketahui mean frekuensi: ", get_frekuensi)
        st.write("Jumlah Transaksi: ", get_len_transaction)
        st.write("minimum support = mean frekuensi/jumlah transaksi")
        st.write("minimum support = {}/{}".format(get_frekuensi, get_len_transaction))
        # Menghitung persentase
        min_support = get_frekuensi / get_len_transaction
        st.write("minimum support = {}".format(min_support))
        min_support = round(get_frekuensi / len(transactions_list), 2)
        st.markdown(
            "<p >nilai minimum support dibulatkan keatas dan dibulatkan 2 angka dibelakang koma</p>", unsafe_allow_html=True)
        st.write("minimum support = {}".format(min_support))

        # MENAMPILKAN MODELING MIN CONFIDENCE
        st.markdown(
            "<h3 >Minimum Confidence</h3>", unsafe_allow_html=True)
        st.markdown(
            "<p >nilai minimum confidence ditetapkan 70 persen berdasarkan referensi penelitian</p>", unsafe_allow_html=True)
        min_confidence = 0.7
        st.write(min_confidence)

        #  MENAMPILKAN MODELING Frequent itemsets
        st.markdown("<h3>Frequent Itemsets</h3>", unsafe_allow_html=True)
        st.dataframe(session_state.df_association_unique, use_container_width=True)

        #  MENAMPILKAN MODELING Rules
        st.markdown("<h3>Rules</h3>", unsafe_allow_html=True)
        st.write("terdapat = {} rules".format(session_state.rules.shape[0]))
        st.dataframe(session_state.rules, use_container_width=True)

        #  MENAMPILKAN EVALUASI RULES
        st.markdown("<h2>Evaluasi</h2>",unsafe_allow_html=True)
        st.markdown("<h3>Menggabungkan antecedents dan consequents</h3>",unsafe_allow_html=True)
        st.dataframe(session_state.df_association, use_container_width=True)
        st.markdown("<h3>Rekomendasi produk bundling</h3>",unsafe_allow_html=True)
        st.markdown("<p>Rekomendasi produk bundling adalah produk yang didapatkan dari rules yang telah dilakukan penghapusan dulikasi rules, nantinya rekomendasi produk ini akan digabungkan dengan proses pembuatan produk bundling yang dilakukan Greenspaces.id dan akan menghasilkan rekomendasi paket bundling dan rekomendasi harga</p>",unsafe_allow_html=True)
        st.dataframe(session_state.df_association_unique, use_container_width=True)

class PaketBundling:
    def pilihKategori(list_produk):
        if 'df_pilih_kategori' not in session_state:
            session_state.df_pilih_kategori = None

        # Mendapatkan kategori dari kalimat pertama pada 'Lineitem name'
        df_category_produk = pd.DataFrame({'Category': list_produk['Lineitem name'].str.split(' ', n=1).str[0]})
        df_category_produk.drop_duplicates(inplace=True)

        # Deleting rows containing 'RP' in the 'Category' column
        df_category_produk = df_category_produk[~df_category_produk['Category'].str.contains('RP')]

        # Mereset index
        df_category_produk.reset_index(drop=True, inplace=True)

        # Mendapatkan daftar kategori unik dari DataFrame
        kategori_list = df_category_produk['Category'].unique()

        st.markdown(
            "<h3>List Produk</h3>", unsafe_allow_html=True)
        list_produk_katergori = list_produk.copy()
        list_produk_katergori['Kategori'] = list_produk_katergori['Lineitem name'].str.split(' ', n=1).str[0]
        list_produk_kategori_select = st.selectbox('Filter Kategori List Produk', kategori_list, key="list_produk_kategori_select")
        list_produk_katergori_show = list_produk_katergori[list_produk_katergori['Lineitem name'].str.contains(list_produk_kategori_select)]
        st.dataframe(list_produk_katergori_show, use_container_width=True)

        st.markdown(
            "<h3>Pilih produk berdasakan kategori</h3>", unsafe_allow_html=True)

        # Menampilkan select option dengan opsi kategori
        kategori_banyak_terjual = st.selectbox('Paling banyak terjual', kategori_list, key="select_banyak_terjual")
        
        df_kategori_tertinggi = list_produk[list_produk['Lineitem name'].str.contains(kategori_banyak_terjual)]
        df_kategori_tertinggi = df_kategori_tertinggi.nlargest(1, 'Support')
        # df_kategori_tertinggi.drop(columns=["Support","Category"], inplace=True)

        # Menampilkan select option dengan opsi kategori
        kategori_sedikit_terjual = st.selectbox('Paling sedikit terjual', kategori_list, key="select_sedikit_terjual")

        df_kategori_terendah = list_produk[list_produk['Lineitem name'].str.contains(kategori_sedikit_terjual)]
        df_kategori_terendah = df_kategori_terendah.nsmallest(1, 'Support')
        # df_kategori_terendah.drop(columns=["Support",'Category'], inplace=True)
        
        df_kategori = pd.concat([df_kategori_tertinggi, df_kategori_terendah], ignore_index=True)
        session_state.df_pilih_kategori = df_kategori
        
        st.markdown("<h5>Produk yang dipilih berdasarkan kategori</h5>", unsafe_allow_html=True)
        st.dataframe(df_kategori, use_container_width=True)
        return df_kategori
    
    def inputTanamanBaru():
        # Inisialisasi session_state jika belum ada
        if 'data_tanaman_baru' not in session_state:
            session_state.data_tanaman_baru = []
        
        st.markdown("<h3>Masukkan tanaman yang ingin di bundling</h3>", unsafe_allow_html=True)
        st.markdown("<p>Masukkan tanaman baru atau tanaman yang diinginkan serta harga jualnya untuk ditambahkan sebagai produk bundling (Opsional)</p>", unsafe_allow_html=True)
        colTanaman, colHarga = st.columns(2)
        # Tampilkan input nama dan harga tanaman
        with colTanaman:
            nama_tanaman = st.text_input("Masukkan Nama Tanaman:")
        with colHarga:
            harga_tanaman = st.number_input("Masukkan Harga Jual Tanaman:", step=1)

        # Tombol untuk menambahkan tanaman ke daftar
        if st.button('Tambah Tanaman',key="tambahtanaman", use_container_width=True  ):
            if nama_tanaman != "" and harga_tanaman > 0:
                # Periksa apakah nama tanaman sudah ada dalam daftar
                nama_tanaman_ada = any(tanaman['Lineitem name'] == nama_tanaman for tanaman in session_state.data_tanaman_baru)
                nama_tanaman_exist_in_kategori = nama_tanaman in session_state.df_pilih_kategori['Lineitem name'].values
                if not nama_tanaman_exist_in_kategori:
                    if not nama_tanaman_ada:
                        tanaman = {'Lineitem name': nama_tanaman, 'Lineitem price': harga_tanaman}
                        session_state.data_tanaman_baru.append(tanaman)
                    else:
                        st.error(f"Produk sudah dimasukkan")
                else:
                    st.error(f"Produk yang ingin dimasukkan sudah dipilih pada pemilihan produk berdasarkan kategori")
            else:
                st.error(f"Nama dan Harga Jual produk tidak boleh kosong!")

        # Menampilkan list tanaman yang sudah dimasukkan
        if len(session_state.data_tanaman_baru) > 0:
            for idx, tanaman in enumerate(session_state.data_tanaman_baru):
                st.write(f"{idx + 1}. {tanaman['Lineitem name']} - Harga: {tanaman['Lineitem price']}")

                # Tombol "Delete" untuk menghapus tanaman dari daftar
                if st.button(f"Delete {tanaman['Lineitem name']}", key=f"delete_{idx}"):
                    session_state.data_tanaman_baru.pop(idx)
                    st.experimental_rerun()

    def buatPaketBundling():
        if st.button("Buat Paket Bundling", type='primary', use_container_width=True, key="buat_paket_bundling"):
            if session_state.df_pilih_kategori is not None:
                if not session_state.df_pilih_kategori.duplicated().any():
                    if len(session_state.data_tanaman_baru) > 0 :
                        # Inisialisasi session_state jika belum ada
                        df_data_tanaman_baru = pd.DataFrame(session_state.data_tanaman_baru)
                        df_data_tanaman_baru['Harga Beli'] = (df_data_tanaman_baru['Lineitem price'] / 4).round(2)
                        df_data_tanaman_baru['Harga Minimum Jual'] = (df_data_tanaman_baru['Harga Beli'] * 3).round(2)
                        df_data_tanaman_baru['Support'] = 0
                        
                        df_bundling_now = pd.concat([session_state.df_pilih_kategori, df_data_tanaman_baru], ignore_index=True)
                        if not df_bundling_now.duplicated().any():
                            if 'list_produk_new' not in session_state:
                                session_state.list_produk_new = pd.DataFrame

                            if 'df_bundling_now' not in session_state:
                                session_state.df_bundling_now = pd.DataFrame

                            list_produk_new  = pd.concat([session_state.list_produk, df_data_tanaman_baru], ignore_index=True)
                            session_state.list_produk_new = list_produk_new
                            session_state.df_bundling_now = df_bundling_now
                            session_state.selected_page = "Hasil Paket Bundling"
                            st.experimental_rerun()
                        else:
                            st.error("Terdapat duplikat produk pada pilih kategori dengan tanaman yang ingin di bundling")
                    else:
                        st.error("Produk yang ingin dimasukkan tidak boleh kosong") 
                else:
                    st.error("Produk tidak boleh ada yang sama pada pemilihan produk berdasarkan kategori")

    def menyimpanDataPaketBundling(df):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='Sheet1', index=False)
        writer.close()  # Menutup objek writer
        output.seek(0)
        data_bundling_download = output.read()
        
        st.markdown("<h4>Unduh hasil pembuatan paket bundling</h4>", unsafe_allow_html=True)
        st.download_button(
            label="Unduh Data",
            data=data_bundling_download,
            file_name='rekomendasi-bundling.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            key='download-button',  # Tambahkan kunci unik jika ada masalah
            use_container_width=True
        )
    
    def tampilPaketBundling(df_dict):
        # Mengakses DataFrame untuk masing-masing Kode Rules menggunakan perulangan
        for kode_rules, rekomendasi_bundling in df_dict.items():
            st.write(f"Rekomendasi ke {kode_rules}")
            rekomendasi_bundling_show = rekomendasi_bundling.reset_index(drop=True)
            rekomendasi_bundling_show = rekomendasi_bundling.drop(columns=['Kode Rules'])
            rekomendasi_bundling_show.index = range(1, len(rekomendasi_bundling_show) + 1)

            # Tambahkan kolom 'no' dengan nilai dari index
            rekomendasi_bundling_show['No'] = rekomendasi_bundling_show.index
            rekomendasi_bundling_show = rekomendasi_bundling_show.set_index('No')
            rekomendasi_bundling_show_table = rekomendasi_bundling_show[['Lineitem name', 'Lineitem price', 'Harga Beli', 'Harga Minimum Jual']]
            st.dataframe(rekomendasi_bundling_show_table, use_container_width=True)
            # Menghitung nilai total dari kolom "Nilai"
            # jumlah_harga_beli = rekomendasi_bundling["Harga Beli"].sum()
            # rekomendasi_harga_bundling =  jumlah_harga_beli * 3
            # total_keuntungan = rekomendasi_harga_bundling - jumlah_harga_beli
            st.success(f"Rekomendasi harga bundling '{rekomendasi_bundling_show['Rekomendasi Harga Bundling'].iloc[0]}'")
            st.success(f"Keuntungan '{rekomendasi_bundling_show['Keuntungan'].iloc[0]}'")
            
            if rekomendasi_bundling_show.shape[0] > 10:
                st.error("Jumlah produk melebih maksimum, paket bundling tidak boleh lebih dari 10")
            st.divider()
