import streamlit as st
from streamlit import session_state
from streamlit_option_menu import option_menu
# Import the function from the module
from module import AssociationRules as ar, PaketBundling as pb
import pandas as pd


class AntarmukaTampilan:
    # Page 1 content
    def halamanMasukkanTranasksi():
        # Menampilkan judul yang rata tengah
        st.markdown(
            "<h1 >Selamat Datang di Sistem Rekomendasi Produk Bundling</h1>", unsafe_allow_html=True)
        st.divider()

        upload = ar.memasukkanTransaksi()
        ar.validasiMasukkanTransaksi(upload)

        if st.button('Lihat proses data mining', type="primary", use_container_width=True):
            if 'upload_transactions' in session_state:
                if not ar.getUploadTransaksi().empty:
                    session_state.selected_page = "Proses Association Rules"
                    st.experimental_rerun()
                else:
                    st.error("Masukkan file transaksi terlebih dahulu")

    # Page 2 content
    def halamanProsesAssociationRule():
        st.markdown(
            "<h1>Halaman Proses Association Rules</h1>", unsafe_allow_html=True)
        st.markdown("<p>Pada halaman ini akan ditampilkan proses association rules yang terdiri dari pemilihan attribute, cleaning data, transform data dan pengimplementasian association rule untuk mendapatkan produk rekomendasi</p>", unsafe_allow_html=True)
        st.divider()
        if not ar.getUploadTransaksi().empty:
            df_transactions= ar.getUploadTransaksi()
            selectAttribute = ar.selectAttribute(df_transactions)
            transformText = ar.transformText(selectAttribute)
            dataCleaning = ar.dataCleaning(transformText)
            transactions_list, trans_encoder_matrix = ar.transformData(dataCleaning)
            list_produk = ar.createListProduk(dataCleaning)
            min_support = ar.minSupport(list_produk, transactions_list) 
            ar.tampilProsesAssociationRules()
            ar.rules(trans_encoder_matrix, min_support)
        else:
            st.error("Masukkan file transaksi terlebih dahulu")
            if st.button("Upload Data",  type="primary", use_container_width=True):
                session_state.selected_page = 'Masukkan Data Transaksi'
                st.experimental_rerun()

    # Page 3 content
    def halamanBuatPaketBundling():
        st.markdown(
            "<h1>Halaman Buat Paket Bundling</h1>", unsafe_allow_html=True)
        # st.markdown("<p>Lorem ipsum dolor sit amet</p>", unsafe_allow_html=True)
        st.divider()
        # pb.associationRulesDev()
        if ar.MemeriksaUploadTransaksi():
            if ar.memeriksaRules():
                if ar.memeriksaListProduk():
                    pb.pilihKategori(ar.getListProduk())
                    pb.inputTanamanBaru()
                    pb.buatPaketBundling()
                else:
                    st.error("List produk tanaman hias tidak ada")
            else:
                st.error("Tidak dapat membuat paket bundling karena tidak terdapat rekomendasi produk bundling. Masukkan file transaksi dengan rentang waktu yang lain")
        else:
            st.error("Masukkan file transaksi terlebih dahulu")
            if st.button("Upload Data",  type="primary",use_container_width=True):
                session_state.selected_page = 'Masukkan Data Transaksi'
                st.experimental_rerun()

    # Page 4 content
    def halamanHasilPaketBundling():
        st.markdown(
            "<h1>Halaman Hasil Paket Bundling</h1>", unsafe_allow_html=True)
        if session_state.get('df_bundling_now') is not None:
            st.balloons()

            df_bundling_now = session_state.df_bundling_now
            list_produk_new = session_state.list_produk_new

            df_bundling_now_list = df_bundling_now['Lineitem name'].tolist()
            # Gunakan metode 'applymap' untuk menerapkan fungsi lambda pada setiap sel di dalam kolom 'Produk Rules'
            df_association_unique = session_state.df_association_unique
            df_association_unique['Produk Rules'] = df_association_unique['Produk Rules'].apply(lambda x: x + df_bundling_now_list)
            
            # Gunakan apply dengan lambda untuk menghilangkan nilai duplikat dari setiap list
            df_association_unique['Produk Rules'] = df_association_unique['Produk Rules'].apply(lambda x: list(set(x)))
            
            # Create a new DataFrame with 'Nama Tanaman' and 'Kode Rules'
            nama_tanaman = []
            kode_rules = []

            for idx, row in df_association_unique.iterrows():
                for plant in row['Produk Rules']:
                    nama_tanaman.append(plant)
                    kode_rules.append(idx + 1)

            list_rules_produk = pd.DataFrame({'Lineitem name': nama_tanaman, 'Kode Rules': kode_rules})

            bundling_with_price = pd.merge(list_rules_produk, list_produk_new, on='Lineitem name', how='left')
            # bundling_with_price = bundling_with_price.drop(columns=['Frekuensi'])

            bundling_download =  bundling_with_price.copy()
            
            # Menggabungkan berdasarkan 'Kode Rules' dan menghitung jumlah total 'Harga Beli' untuk setiap Kode Rules
            total_harga_beli = bundling_download.groupby('Kode Rules')['Harga Beli'].sum()

            # Menambahkan kolom baru 'Rekomendasi Harga Bundling' yang dihitung dengan mengalikan jumlah total 'Lineitem price' dengan 3
            bundling_download['Rekomendasi Harga Bundling'] = bundling_download['Kode Rules'].map(
                total_harga_beli) * 3
            
            # Membuat kolom 'keuntungan' berdasarkan selisih antara 'Rekomendasi Harga Bundling' dan 'Harga Beli'
            bundling_download['Keuntungan'] = bundling_download['Rekomendasi Harga Bundling'] - bundling_download['Kode Rules'].map(total_harga_beli)

            
            # Membuat dictionary berdasarkan 'Kode Rules' dengan masing-masing DataFrame
            df_dict = dict(
                tuple(bundling_with_price.groupby('Kode Rules')))

            # pb.menyimpanDataPaketBundling(bundling_download)
            st.markdown("<h4>Rekomendasi Paket Bundling</h4>", unsafe_allow_html=True)
            st.markdown("<p>Harga beli didapatkan dari harga jual yang terdapat pada kolom Lineitem price dibagi 4. harga beli merupakan harga modal perusahaan terhadap produk</p>", unsafe_allow_html=True)
            st.markdown("<p>Rekomendasi harga paket bundling didapatkan dari total harga beli produk x 3</p>", unsafe_allow_html=True)
            st.write("Terdapat ", len(df_dict), ' rekomendasi paket bundling')
            pb.tampilPaketBundling(df_dict)
        else:
            st.divider()
            st.error("Paket bundling belum dibuat")
            if st.button("Buat Paket Bundling",  type="primary",use_container_width=True):
                session_state.selected_page = 'Buat Paket Bundling'
                st.experimental_rerun()

# Main function to run the app
class Main():
    def main():
        if 'selected_page' not in session_state:
                session_state.selected_page = 'Masukkan Data Transaksi'

        st.sidebar.title("Navigasi")
        ui = AntarmukaTampilan
        pages = {
            "Masukkan Data Transaksi": ui.halamanMasukkanTranasksi,
            "Proses Association Rules": ui.halamanProsesAssociationRule,
            "Buat Paket Bundling": ui.halamanBuatPaketBundling,
            "Hasil Paket Bundling": ui.halamanHasilPaketBundling
        }

        # Display buttons for each page in the sidebar
        selected_page = st.sidebar.button("Masukkan Data Transaksi", use_container_width=True)
        if selected_page:
            session_state.selected_page = "Masukkan Data Transaksi"

        selected_page = st.sidebar.button("Proses Association Rules", use_container_width=True)
        if selected_page:
            session_state.selected_page = "Proses Association Rules"

        selected_page = st.sidebar.button("Buat Paket Bundling", use_container_width=True)
        if selected_page:
            session_state.selected_page = "Buat Paket Bundling"

        selected_page = st.sidebar.button("Hasil Paket Bundling", use_container_width=True)
        if selected_page:
            session_state.selected_page = "Hasil Paket Bundling"

        # Execute the selected page function
        pages[session_state.selected_page]()

if __name__ == "__main__":
    Main.main()
