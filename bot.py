from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import textwrap

model_name = "mrizalf7/indobert-qa-finetuned-small-squad-indonesian-rizal"

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Fungsi untuk bagi dokumen panjang jadi potongan
def chunk_context(text, max_length=500):
    return textwrap.wrap(text, max_length)

# Load konteks panjang dari file atau langsung dari string
context =  """
Indonesia, dengan nama resmi Republik Indonesia,[a] adalah sebuah negara kepulauan di Asia Tenggara yang dilintasi garis khatulistiwa dan berada di antara daratan benua Asia dan Oseania sehingga dikenal sebagai negara lintas benua, serta antara Samudra Pasifik dan Samudra Hindia.

Indonesia merupakan negara terluas ke-14 sekaligus negara kepulauan terbesar di dunia dengan luas wilayah sebesar 1.904.569 km²,[10] serta negara dengan pulau terbanyak ke-6 di dunia, dengan jumlah 17.504 pulau.[11] Nama alternatif yang dipakai untuk kepulauan Indonesia disebut Nusantara.[12] Selain itu, Indonesia juga menjadi negara berpenduduk terbanyak ke-4 di dunia dengan penduduk mencapai 275.344.166 jiwa pada tahun 2022.[13] Indonesia adalah negara multiras, multietnis, dan multikultural di dunia, seperti halnya Amerika Serikat.[14] Indonesia berbatasan dengan sejumlah negara di Asia Tenggara dan Oseania. Indonesia berbatasan di wilayah darat dengan Malaysia di Pulau Kalimantan dan Sebatik, dengan Papua Nugini di Pulau Papua, dan dengan Timor Leste di Pulau Timor. Negara yang hanya berbatasan laut dengan Indonesia adalah Singapura, Filipina, Australia, Thailand, Vietnam, Palau, dan wilayah persatuan Kepulauan Andaman dan Nikobar, India.

Indonesia adalah negara kesatuan dengan bentuk pemerintahan republik berdasarkan konstitusi yang sah, yaitu Undang-Undang Dasar Negara Republik Indonesia Tahun 1945 (UUD 1945).[15] Berdasarkan UUD 1945 pula, Dewan Perwakilan Rakyat (DPR), Dewan Perwakilan Daerah (DPD), dan Presiden dicalonkan lalu dipilih dalam pemilihan umum. Ibu kota Indonesia saat ini adalah Jakarta. Pada tanggal 18 Januari 2022, pemerintah Indonesia menetapkan Ibu Kota Nusantara yang berada di Pulau Kalimantan, yang menempati wilayah Kabupaten Penajam Paser Utara, untuk menggantikan Jakarta sebagai ibu kota yang baru.[16] Hingga tahun 2022, proses peralihan ibu kota masih berlangsung.

Sejarah Indonesia banyak dipengaruhi oleh bangsa-bangsa pendatang dan penjajah. Kepulauan Indonesia menjadi wilayah perdagangan penting sejak abad ke-7, yaitu sejak berdirinya Sriwijaya, kerajaan bercorak Hinduisme-Buddhisme yang berpusat di Palembang. Kerajaan Sriwijaya menjalin hubungan agama dan perdagangan dengan bangsa Tionghoa, India, dan juga Arab. Agama dan kebudayaan Hinduisme-Buddhisme tumbuh, berkembang, dan berasimilasi di kepulauan Indonesia pada awal abad ke-4 hingga abad ke-13 Masehi. Setelah itu, para pedagang sufi dan Islam sunni membawa agama dan kebudayaan Islam sekitar abad ke-8 hingga abad ke-16. Pada akhir abad ke-15, bangsa-bangsa Eropa datang ke kepulauan Indonesia dan berperang untuk memonopoli perdagangan rempah-rempah di Maluku semasa Zaman Penjelajahan. Setelah berada di bawah kolonial Belanda, Indonesia yang saat itu bernama Hindia Belanda, memproklamasikan kemerdekaan di akhir Perang Dunia II, tepatnya pada tanggal 17 Agustus 1945. Selanjutnya, Indonesia mendapat berbagai tantangan dan persoalan berat, mulai dari bencana alam, praktik korupsi yang masif, konflik sosial, gerakan separatisme, proses demokratisasi, dan periode pembangunan, perubahan dan perkembangan sosial–ekonomi–politik, serta modernisasi yang pesat.


Batas garis pangkal kepulauan dari wilayah Negara Kesatuan Republik Indonesia berdasarkan artikel 47, paragraf 9 UNCLOS
Indonesia terdiri dari berbagai suku bangsa, bahasa, dan agama. Berdasarkan rumpun bangsa, Indonesia terdiri atas bangsa asli pribumi yakni Austronesia dan Melanesia di mana bangsa Austronesia yang terbesar jumlahnya dan lebih banyak mendiami Indonesia bagian barat. Dengan suku Jawa dan Sunda membentuk kelompok suku bangsa terbesar dengan persentase mencapai 57% dari seluruh penduduk Indonesia.[17] Semboyan nasional Indonesia, "Bhinneka Tunggal Ika" (Berbeda-beda tetapi tetap satu), bermakna keberagaman sosial-budaya yang membentuk satu kesatuan negara. Selain memiliki penduduk yang padat dan wilayah yang luas, Indonesia memiliki alam yang mendukung tingkat keanekaragaman hayati terbesar ke-2 di dunia.

Etimologi
Lihat pula: Sejarah nama Indonesia
Kata "Indonesia" berasal dari bahasa Yunani kuno yaitu Indus yang merujuk kepada Sungai Indus di India dan nesos yang berarti "pulau".[18] Jadi, kata Indonesia berarti wilayah "kepulauan India", atau kepulauan yang berada di wilayah Hindia; ini merujuk kepada persamaan antara dua bangsa tersebut (India dan Indonesia).[19] Pada tahun 1850, George Windsor Earl, seorang etnolog berkebangsaan Inggris, awalnya mengusulkan istilah Indunesia dan Malayunesia untuk penduduk "Kepulauan Hindia atau Kepulauan Melayu".[20] Murid Earl, James Richardson Logan, menggunakan kata Indonesia sebagai sinonim dari Kepulauan India.[21][22] Namun, penulisan akademik Belanda di media Hindia Belanda tidak menggunakan kata Indonesia, tetapi istilah Kepulauan Melayu (Maleische Archipel); Hindia Timur Belanda (Nederlandsch Oost Indië), atau Hindia (Indië); Timur (de Oost); dan bahkan Insulinde (istilah ini diperkenalkan tahun 1860 dalam novel Max Havelaar (1859) yang ditulis oleh Multatuli mengenai kritik terhadap kolonialisme Belanda).[12]

Sejak tahun 1900, nama Indonesia menjadi lebih umum pada lingkungan akademik di luar Belanda, dan golongan nasionalis Indonesia menggunakannya untuk ekspresi politik.[12] Adolf Bastian dari Universitas Berlin memasyarakatkan nama ini melalui buku Indonesien oder die Inseln des Malayischen Archipels, 1884–1894. Pelajar Indonesia pertama yang menggunakannya ialah Suwardi Suryaningrat (Ki Hadjar Dewantara), ketika ia mendirikan kantor berita di Belanda yang bernama Indonesisch Pers Bureau pada tahun 1913.[19]

Sejarah
Artikel utama: Sejarah Indonesia dan Sejarah Nusantara
Periode prasejarah
Artikel utama: Prasejarah Indonesia

Peta wilayah Sundaland, Sahul, dan Wallacea pada kala Pleistosen.
Kepulauan Indonesia terbentuk melalui berbagai aktivitas tektonis yang sangat kompleks sejak awal masa Senozoikum (sekitar 66 juta tahun lalu) dan mulai mencapai bentuknya yang sekarang ketika memasuki kala Pleistosen (sekitar 2,58 juta tahun lalu).[23] Pada kala tersebut, permukaan laut global saat itu rata-rata lebih rendah 130 meter daripada permukaan laut global sekarang,[24] sehingga muncul Daratan Sunda (Sundaland) yang terhubung dengan daratan utama Asia dan saat ini mencakup Sumatra, Jawa, Kalimantan, dan lautan-lautan di antaranya,[25][26] serta Benua Sahul yang saat ini mencakup Pulau Papua, Australia, dan Laut Arafura.[27][28] Kedua daratan tersebut diantarai oleh Kepulauan Wallacea yang saat ini mencakup Sulawesi, Nusa Tenggara, dan Maluku.[29] Sekitar 74.000 tahun yang lalu, letusan dahsyat berskala VEI-8 terjadi pada Gunung Toba (sekarang menjadi Danau Toba). Letusan tersebut konon menjadi letusan gunung berapi terbesar yang berhasil diteliti. Perubahan iklim yang ditimbulkannya diperkirakan menjadi penyebab populasi manusia modern dunia hampir seluruhnya musnah dan pergerakan migrasi manusia sempat terhenti pada subkala Pleistosen Akhir.[30][31] Lalu pada akhir periode glasial terakhir (sekitar 12.000 tahun lalu), permukaan laut naik setinggi 60 meter hanya dalam kurun waktu lima milenium.[32] Akibatnya, daratan yang lebih rendah terendam dan membentuk perairan dangkal, sementara daratan yang lebih tinggi terpisah-pisah menjadi pulau-pulau yang lebih kecil. Pulau-pulau tersebut membentuk kepulauan Indonesia seperti sekarang ini.[33]


Ilustrasi "Manusia Jawa" oleh J. H. McGregor.
Dari kumpulan fosil manusia purba Homo erectus (atau manusia Jawa) dan Homo floresiensis ("manusia Flores") yang pernah menetap di Indonesia, kuat dugaan bahwa kepulauan Indonesia telah dihuni oleh manusia purba tersebut sekurang-kurangnya antara dua juta sampai 500.000 tahun yang lalu. Manusia purba tersebut kemudian berangsur-angsur punah seiring dengan kedatangan manusia modern (Homo sapiens) di kepulauan Indonesia.[34][35]

Gelombang migrasi manusia modern pertama kali sampai di kepulauan Indonesia melalui jalur darat sekitar 60.000 tahun yang lalu. Gelombang pertama ini menjadi nenek moyang dari bangsa Melanesia.[36][37] Kemudian sekitar 3.500–1.500 SM, bangsa Austronesia yang berasal dari Taiwan tiba melalui jalur laut dan menetap di kepulauan Indonesia. Sebagian bangsa Melanesia yang telah ada lebih dahulu terdesak ke wilayah-wilayah timur jauh, sementara sebagian lagi berasimilasi dengan pendatang tersebut.[36][38][39] Manusia yang menetap tersebut kemudian mengembangkan budaya bercocok tanam dan melaut.[40]

Periode monarki
Kerajaan Hindu-Buddha
Artikel utama: Sejarah Nusantara pada era kerajaan Hindu-Buddha

Situs Percandian Batujaya yang berada di Kabupaten Karawang, Jawa Barat. Candi-candi yang ada di dalamnya merupakan sisa-sisa peninggalan Tarumanagara.
Kandis diduga merupakan kerajaan tertua di Nusantara (kepulauan Indonesia) yang berdiri pada abad ke-1 SM dan terletak di daerah yang saat ini menjadi wilayah Provinsi Riau, tetapi keberadaannya masih sering diperdebatkan oleh para sejarawan, karena tidak adanya bukti yang jelas atas kerajaan ini.[41] Keberadaan Salakanagara yang berdiri pada abad ke-1 Masehi di daerah sekitar Cianjur, Jawa Barat juga masih menjadi perdebatan oleh para ahli karena kurangnya bukti-bukti sejarah, meskipun kerajaan ini merupakan cikal bakal Tarumanagara.[41]

Dua kerajaan tertua Nusantara yang memiliki bukti-bukti sejarah adalah Kutai Martapura di wilayah Kalimantan Selatan saat ini dan Tarumanagara di wilayah barat Pulau Jawa, yang sama-sama berdiri pada abad ke-4 Masehi.[42] Kedua kerajaan tersebut dibuktikan memiliki corak Hindu-Buddha, sehingga dapat dipastikan bahwa Agama Hindu dan Agama Buddha telah berkembang di Nusantara sekurang-kurangnya dari abad ke-4 M.[43] Banyak kerajaan bercorak Hindu-Buddha lainnya yang kemudian terbentuk setelah itu.


Perkembangan wilayah kekuasaan Sriwijaya sejak berdiri hingga keruntuhannya.
Sriwijaya, yang berbentuk kedatuan dan bercorak Buddha, berdiri di Nusantara pada abad ke-7 Masehi, kemudian berkembang menjadi salah satu kemaharajaan terbesar di Nusantara, serta negara monarki dengan masa berdiri terlama di Asia Tenggara.[44] Pada masa kejayaannya, Sriwijaya melingkupi Sumatra, Malaya, Kra, Jawa, Kalimantan, Kamboja, dan Vietnam,[45][46] serta berkuasa dalam mengendalikan aktivitas pelayaran dan perdagangan di Selat Malaka yang merupakan jalur pelayaran penting di dunia. Banyak budaya asing yang mempengaruhi dan berasimilasi dengan budaya-budaya lokal.[47] Sejak diperintah oleh Balaputradewa pada pertengahan abad ke-9, Sriwijaya juga berada di bawah kekuasaan Wangsa Sailendra.[48] Nama Sriwijaya diperkirakan mulai meredup dan runtuh pada awal abad ke-11 dan digantikan oleh Dharmasraya, lalu oleh Pagaruyung pada abad ke-14.[49]

Medang, yang diperintah oleh Wangsa Sailendra, berdiri di wilayah Jawa Tengah saat ini pada abad ke-8.[50][51] Pada abad ke-10, pusat pemerintahannya dipindahkan ke Jawa Timur dan para penguasa setelah kepindahan tersebut dikelompokkan dalam Wangsa Isyana.[52] Pada tahun 1016, Medang runtuh akibat pemberontakan yang menewaskan raja terakhir beserta banyak kerabatnya.[53] Airlangga, menantu raja tersebut, membangun ulang kerajaan dan mendirikan negara Kahuripan pada tahun 1019,[54] yang kemudian terpecah menjadi Kadiri dan Janggala pada tahun 1042. Janggala lalu ditaklukkan oleh Kadiri pada tahun 1135. Ken Arok dari Wangsa Rajasa kemudian menaklukkan Kadiri dan mendirikan Singasari pada tahun 1222. Singasari runtuh pada tahun 1292 akibat pemberontakan yang dipimpin oleh Jayakatwang (sisa Wangsa Isyana), tetapi berhasil ditumpas setahun kemudian oleh Raden Wijaya.[51][52]


Perkembangan wilayah kekuasaan Majapahit sejak berdiri hingga keruntuhannya.
Raden Wijaya dari Wangsa Rajasa mendirikan Majapahit yang bercorak Syiwa-Buddha pada tahun 1293, yang kemudian berkembang menjadi kemaharajaan terbesar di Nusantara dan juga di Asia Tenggara, serta menjadi negara agraris dan jalur perdagangan dunia.[55] Majapahit mencapai masa kejayaannya pada masa kejayaannya pemerintahan Hayam Wuruk dengan patihnya, Gajah Mada (terkenal dengan sunpahnya yang bernama Sumpah Palapa),[55] dengan wilayah kekuasaan meliputi Sumatra, Malaya, Kalimantan, Sulawesi, Nusa Tenggara, Maluku, hingga Papua.[52] Majapahit mengalami kemunduran seiring menguatnya pengaruh Islam di Nusantara, lalu akhirnya runtuh setelah ditaklukkan oleh Demak pada tahun 1527.

Pengaruh Hindu-Buddha semakin berkurang seiring dengan masuknya Islam di Nusantara. Namun, beberapa kerajaan bercorak Hindu-Buddha masih bertahan bahkan hingga kolonialisme masuk di Nusantara, seperti Blambangan di Pulau Jawa,[56] serta kerajaan-kerajaan Bali bekas Gelgel, yakni Klungkung, Buleleng, Karangasem, Badung, Tabanan, Gianyar, Bangli, Mengwi, dan Jembrana.[57]

Kesultanan Islam
Artikel utama: Sejarah Nusantara pada era kerajaan Islam

Bendera Aceh, kesultanan lampau terbesar di Sumatra.
Islam mulai dibawa masuk ke Nusantara oleh para pedagang dan para ulama berkebangsaan Arab, Persia, Gujarat, dan Tionghoa pada abad ke-7 Masehi.[58][59] Aceh menjadi pusat penyebaran agama Islam pertama di Nusantara,[60] serta menjadi lokasi negara kesultanan pertama yang pernah berdiri di Nusantara, yaitu negara Jeumpa yang berdiri pada abad ke-7 dan menguasai wilayah Kabupaten Bieruen saat ini.[61] Setelah Sriwijaya runtuh pada abad ke-11, Islam mulai menyebar ke berbagai daerah di Sumatra dan membuat beberapa kerajaan Hindu-Buddha di Sumatra beralih menjadi kesultanan Islam. Aceh (berdiri pada tahun 1496) menjadi kesultanan terbesar di Pulau Sumatra yang mencapai masa kejayaannya di bawah perintah Iskandar Muda (1607–1636).[62]


Bendera Mataram, salah satu kesultanan terbesar di Jawa.
Islam mulai diperkenalkan dan menyebar secara luas di kepulauan Indonesia lainnya pada abad ke-15.[63] Setelah keruntuhan Majapahit, kesultanan-kesultanan Islam Nusantara mulai berdiri dan berkembang pesat. Lumajang (berdiri pada akhir abad ke-13) diperkirakan merupakan kesultanan Islam yang paling tua meskipun belum ada bukti-bukti pendukung yang cukup.[64] Kesultanan pertama di Pulau Jawa yang dapat dibuktikan oleh para sejarawan adalah Demak dan Cirebon, yang sama-sama berdiri pada abad ke-15 dan menjadi salah satu negara terbesar di Jawa.[65][66] Mataram, yang didirikan pada tahun 1586 oleh Wangsa Mataram, juga menjadi salah satu negara berpengaruh di Jawa, sebelum akhirnya terpecah melalui Perjanjian Giyanti.[67][68]


Bendera Banjar, salah satu kesultanan terbesar di Kalimantan.
Beberapa kesultanan baru mulai berdiri di Kalimantan sejak abad ke-14 seiring dengan meningkatnya pengaruh Islam, bahkan beberapa kerajaan Hindu-Buddha beralih menjadi kesultanan. Brunei berhasil mencapai masa kejayaannya pada abad ke-15 setelah menguasai seluruh pesisir Kalimantan.[69] Banjar (berdiri pada tahun 1520) berkembang menjadi salah satu negara terbesar di Pulau Kalimantan setelah menguasai pesisir selatan Kalimantan,[70] sebelum akhirnya merosot pada abad ke-18 dan dihapuskan oleh pemerintah kolonial pada tahun 1905.[71]

Islam diperkirakan berkembang di Sulawesi sejak abad ke-16 dan beberapa kerajaan bercorak Hindu-Buddha atau berkepercayaan tradisional berubah menjadi kesultanan.[72] Kesultanan terbesar di Pulau Sulawesi adalah persekutuan negara Gowa–Tallo, yang disebut Makassar oleh para ahli, yang ketika masa kejayaannya mencakup Sulawesi, Kalimantan, Nusa Tenggara, Kepulauan Maluku, hingga Australia.

Dua kesultanan dengan pengaruh besar di Kepulauan Maluku adalah Ternate dan Tidore, yang berpusat di wilayah Maluku Utara saat ini.[73] Kedua kesultanan ini mencapai puncak kejayaannya pada abad ke-16 berkat perdagangan rempah-rempah, tetapi kemudian mengalami kemunduran semenjak diadu domba oleh bangsa asing dan akhirnya runtuh di tangan VOC.[74]

Kesultanan-kesultanan Islam mulai merosot ketika bangsa-bangsa asing masuk dan menguasai tanah Nusantara. Belanda yang membentuk Hindia Belanda bahkan membubarkan hampir seluruh monarki di wilayah kolonialnya.[75]

Kerajaan Kristen
Artikel utama: Sejarah Nusantara pada era kerajaan Kristen

Bendera Kerajaan Siau, kerajaan dengan corak agama Kristen tertua di Nusantara.[76]
Kekristenan umumnya dibawa oleh para misionaris Barat yang menumpang pada kapal pemerintah kolonial. Katolik awalnya dibawa ke Nusantara oleh bangsa Portugis, sebelum akhirnya sempat dilarang penyebarannya oleh Pemerintah Belanda yang menguasai Hindia Belanda. Setelah Napoleon sempat menguasai Belanda, penyebaran Katolik menjadi lebih leluasa dan misionaris Katolik Belanda melanjutkan misi di Hindia Belanda.[77] Sementara itu, Protestantisme dibawa oleh misionaris Protestan yang juga berasal dari Belanda.[78]

Beberapa kerajaan bercorak Kristen muncul sewaktu para misionaris menyebarkan Kekristenan pada rakyat dan keluarga kerajaan di kawasan tertentu.[79] Kerajaan-kerajaan Kristen yang terbentuk di Nusantara adalah Bolaang Mongondow, Manganitu, Manado, Moro, Siau, Soya, dan Tagulandang, serta Amanatun, Larantuka, dan Sikka yang bercorak Katolik.[80][81]

Periode kolonial
Upaya kolonisasi oleh Portugal
Artikel utama: Imperium Portugal di Nusantara

Peta buatan tahun 1519 yang menunjukkan pulau-pulau di Maluku Utara, yang dipasangkan dengan bendera Portugal saat itu.
Demi mencari rempah-rempah yang sulit didapatkan setelah jalur perdagangannya terputus akibat jatuhnya Konstantinopel ke tangan bangsa Turki Utsmani pada tahun 1453,[82] armada Portugis di bawah kepemimpinan Afonso de Albuquerque melakukan ekspedisi ke timur Eropa hingga sampai di negara Melaka dan memulai sejarah kolonialisme di Nusantara dengan menyerang dan menduduki negara itu.[83][84] Demak yang merasa terancam lalu mengirim armada laut ke Melaka pada tahun 1453 untuk menyerang balik armada Portugis, tetapi usahanya gagal.[84] Pada tahun 1512, Albuquerque mengirimkan armada laut yang dipimpin oleh António de Abreu dan Francisco Serrão menuju Kepulauan Maluku demi memonopoli perdagangan cengkih dan pala[85] Bayanullah (sultan Ternate saat itu) mengizinkan armada Portugis untuk membangun Benteng Kastela dan memonopoli perdagangan rempah-rempah di Ternate dengan imbalan bantuan militer, karena Ternate pada saat itu sedang bermusuhan dengan Tidore.[84]

Armada Spanyol yang melakukan ekspedisi ke barat Eropa melanjutkan ekspedisi di bawah kepemimpinan Juan Sebastián Elcano setelah kehilangan banyak pasukan di Filipina dan akhirnya tiba di Kepulauan Maluku pada tanggal 8 November 1521, tetapi kedatangannya ditentang oleh armada Portugis yang terlebih dahulu ada di sana dan menganggap Spanyol melanggar Perjanjian Tordesillas. Bangsa Spanyol bersekutu dengan Tidore untuk melawan Ternate dan Portugal.[86] Persaingan kubu Ternate–Portugal vs. Tidore–Spanyol berujung pada meletusnya perang antarkubu, yang berakhir dengan kekalahan kubu Tidore–Spanyol dan penandatanganan Perjanjian Zaragoza pada tanggal 22 April 1529, yang membuat armada Spanyol harus angkat kaki dari Maluku dan kembali ke Filipina.[87]


Peta kolonisasi bangsa Portugis di Nusantara.
Sementara itu, armada Portugis ingin meneruskan ambisi memperbesar koloni di Nusantara dengan cara menguasai Selat Sunda dan akhirnya mereka membuat perjanjian dengan Prabu Surawisesa (raja Sunda saat itu) pada tahun 1522, yang mengizinkan pendirian benteng di Banten dan Sunda Kelapa bagi armada Portugis dengan imbalan bantuan militer untuk menghadapi Demak dan Cirebon. Namun, kerja sama tersebut tidak pernah terlaksana, karena armada yang dikirim untuk melaksanakan perjanjian itu terseret dalam badai topan di Teluk Benggala dan beberapa pasukan yang tiba di Sunda dengan selamat diserang oleh pasukan Fatahillah yang sedang menyerbu Sunda, sehingga armada Portugis akhirnya meninggalkan Selat Sunda.[85]

Setelah kepergian Spanyol, bangsa Portugis mulai mencoba untuk memperbesar pengaruh mereka, sementara Ternate mulai menyadari bahwa Portugal sudah terlalu banyak ikut campur urusan internal negara, terutama atas suksesi takhta. Tewasnya Khairun Jamil (sultan Ternate) oleh pasukan Portugis memantik kemarahan rakyat Ternate dan memicu Perang Ternate–Portugal. Ternate dan sekutunya berhasil memenangkan perang dan mengusir sebagian besar pasukan Portugis yang lari menuju Nusa Tenggara.[87] Pengaruh bangsa Portugis di Nusantara semakin berkurang setelah bangsa Belanda mulai masuk ke Nusantara dan akhirnya hanya tersisa di wilayah Pulau Timor bagian timur menurut Perjanjian Lisboa.[88]

Monopoli VOC
Artikel utama: Perusahaan Hindia Timur Belanda di Nusantara

Lambang VOC, suatu serikat dagang Belanda yang memonopoli perdagangan rempah di Nusantara.
Berbekal rute pelayaran armada Portugis sebelumnya, armada kapal Belanda di bawah kepemimpinan Cornelis de Houtman memulai ekspedisi pertamanya untuk mencari rempah-rempah di Timur, hingga akhirnya sampai di Banten pada tanggal 27 Juni 1596, serta berhasil menyusuri pesisir utara Jawa hingga ke Bali dalam kurun waktu setahun. Tabiat buruk Houtman dan anak buahnya membuat mereka sering berseteru dengan penduduk lokal di sepanjang perjalanan, meskipun mereka akhirnya sukses membawa serta peti-peti berisi rempah dalam jumlah banyak kembali Belanda.[89] Pada tahun 1598–1600, para pedangang Belanda membentuk rombongan ekspedisi yang dipimpin oleh Jacob Corneliszoon van Neck agar dapat mengulang kesuksesan tersebut. Mereka berusaha menarik hati para penduduk dan penguasa lokal untuk tidak mengulangi kesalahan yang dilakukan armada Portugis dan rombongan Houtman. Setelah itu, berbagai kapal milik para pedagang Belanda lainnya menyusul untuk memperoleh dan menguasai rempah-rempah di Nusantara.[89]


Peta Asia Tenggara yang dibuat sekitar tahun 1674–1745 oleh Kâtip Çelebi, seorang ahli geografi Turki Utsmani.
Dewan Negara Belanda membentuk suatu serikat dagang pada tanggal 20 Maret 1602bernama Perusahaan Hindia Timur Belanda (VOC) untuk mengurangi persaingan di antara para pedagang rempah Belanda. Dalam piagam "oktroi" (octrooi), VOC diperbolehkan untuk memiliki angkatan perang sendiri, mencetak mata uang sendiri, serta memonopoli perdagangan dan menekan penguasa-penguasa lokal di kawasan Nusantara.[90] Pada tahun 1603, VOC mulai membangun pos-pos perdagangan di Banten, Ambon, Jayakarta, dan lain-lain. Sejak tahun 1604, VOC bersaing ketat dengan armada Perusahaan Hindia Timur Britania (EIC) yang juga tiba di Nusantara demi tujuan yang sama.[91] Pada tanggal 19 Desember 1610, Pieter Both ditunjuk sebagai gubernur jenderal pertama di Nusantara, yang kemudian menetapkan Ambon sebagai pusat pemerintahan.[90] Pada tanggal 30 Mei 1619, Jan Pieterszoon Coen (gubernur jenderal yang baru) memerintahkan armada kapal VOC untuk menyerang Jayapura dan Banten, serta mendirikan Batavia yang kelak menjadi pusat pemerintahan. Pada tahun 1620, VOC dan EIC membuat perjanjian perdagangan rempah-rempah, tetapi hubungan tersebut putus sejak armada Inggris berangsur-angsur meninggalkan wilayah Nusantara setelah terjadinya Pembantaian Amboina terhadap beberapa orang Inggris pada tahun 1623.[92] Istilah "Hindia Belanda" (Nederlandsch-Indië) mulai digunakan di dalam dokumen resmi VOC sejak awal tahun 1620-an.[93] VOC menjadi badan usaha swasta yang sangat sukses selama abad ke-17 dan bahkan menjadi perusahaan terkaya di dunia pada tahun 1669. VOC lihai dalam melakukan politik adu domba antarkerajaan kecil dan memaksa para penguasa lokal untuk menandatangani perjanjian damai (misalnya Perjanjian Painan). VOC saat itu menguasai Pulau Jawa, Painan, Makassar, Manado, Pulau Seram, dan Pulau Buru.[94]


Pembagian Mataram setelah Perjanjian Giyanti (1755) dan Salatiga (1757).
Pasukan Mataram pernah merencanakan penyerbuan ke markas VOC di Batavia sebanyak dua kali pada tahun 1628 dan 1629, tetapi akhirnya gagal karena kekurangan perbekalan.[95] Sebagai gantinya, VOC beberapa kali mencampuri urusan kerajaan di Mataram berkali-kali, seperti membantu dalam perang takhta melawan pasukan Amangkurat III pada tahun 1704–1708, membantu dalam perang takhta melawan kerabat raja yang memberontak pada tahun 1719–1723, serta ikut campur dalam rangkaian konflik antaranggota keluarga kerajaan Mataram pada tahun 1749–1757. Perjanjian Giyanti (13 Februari 1755) dan Perjanjian Salatiga (17 Maret 1757) yang ditandatangani bersama pihak VOC membuat negara Mataram terpecah menjadi beberapa negara baru, yaitu Mangkunagaran, Yogyakarta, dan Surakarta.[96]

Mulai tahun 1730, kejayaan VOC mulai merosot akibat korupsi di tubuh VOC, ketidaksiapan dalam memenuhi permintaan pasar yang berubah, serta pergolakan yang terus-menerus terjadi di Eropa dan di Nusantara.[97] Pergolakan di Nusantara, misalnya, yaitu pembantaian orang-orang Tionghoa di Batavia pada tahun 1740 yang dikenal dengan peristiwa Geger Pacinan, yang kemudian memicu Perang Jawa (1741–1743) dan Perang Kuning (1750).[98] Lalu pada tahun 1771–1772, Perang Bayu pecah di Blambangan dan memakan korban jiwa yang sangat besar dari penduduk lokal dan pasukan VOC.[99] Setelah perang melawan Inggris (1780-1784) berakhir, VOC mengalami krisis finansial yang sangat buruk yang membuatnya hampir tidak dapat beroperasi. VOC diambil alih oleh Bataaf (penerus Belanda) sejak tanggal 1 Maret 1796 untuk mengatasi krisis tersebut, tetapi akhirnya gagal. Pada tanggal 31 Desember 1799, VOC resmi berhenti beroperasi, sementara aset-asetnya (termasuk koloni VOC) diambil oleh Pemerintah Bataaf,[100] sebelum akhirnya jatuh ke tangan Prancis enam tahun kemudian.

Koloni Belanda dalam kendali Prancis
Artikel utama: Jeda kekuasaan Prancis dan Britania di Hindia Belanda § Kekuasaan Prancis (1806–1811)

Potret Gubernur Jenderal Herman Willem Daendels.
Napoleon Bonaparte yang menguasai Prancis pada saat itu membubarkan Bataaf (negara pengekor dari Prancis) dan mendirikan negara boneka Hollandia pada bulan Maret 1806, lalu menunjuk Louis (adiknya) sebagai raja pada tanggal 5 Juni. Louis mengirimkan Herman Willem Daendels berkebangsaan Belanda sebagai Gubernur Jenderal Hindia Belanda dan tiba di Batavia pada tanggal 5 Januari 1808.[101][102] Daendels kemudian menerapkan aturan yang sangat keras dan kebijakan bertangan besi di Hindia Belanda sebagai persiapan menghadapi ancaman Britania Raya. Daendels membangun banyak fasilitas dan benteng pertahanan, seperti Jalan Raya Pos Anyar–Panarukan yang memakan banyak korban dari pekerja Heerendiensten,[103] Benteng Lodewijk di Surabaya, dan Paleis van Daendels (sekarang Gedung AA Maramis) di Batavia. Daendels juga keras terhadap para penguasa lokal dan keluarganya, serta menjadi penyebab jatuhnya negara Banten.[104] Gaya kepemimpinan tersebut tentu saja menimbulkan kesengsaraan pada penduduk lokal, sehingga pemberontakan yang dipimpin oleh Ronggo Prawirodirjo III akhirnya pecah di Pulau Jawa pada tanggal 20 November – 17 Desember 1810, tetapi cepat diredam oleh pasukan Hindia Belanda dan Keraton Yogyakarta.[105] Daendels turun dari jabatannya pada tanggal 15 Mei 1811. Tidak lama kemudian, Britania Raya menyerbu Pulau Jawa dan mengambil alih Hindia Belanda.[106]

Kolonisasi singkat Britania Raya
Artikel utama: Jeda kekuasaan Prancis dan Britania di Hindia Belanda § Kekuasaan Britania (1811–1816)
Armada gabungan Britania Raya dan EIC berangkat menuju Hindia Belanda pada tahun 1809 untuk merebut wilayah tersebut dari Prancis dan aknirnya berhasil menguasai Kepulauan Maluku setahun setelahnya.[107] Pada bulan Agustus 1811, armada Britania mulai menyerbu Pulau Jawa dan menduduki satu per satu pos milik Prancis dan Belanda di Jawa, hingga pasukan Jan Willem Janssens (Gubernur Jenderal Hindia Belanda saat itu) yang lari dari Batavia akhirnya takluk di Salatiga. Pada tanggal 18 September, pihak Belanda menyerahkan kekuasaan atas Hindia Belanda secara resmi kepada armada Britania melalui Perjanjian Tuntang.[108][109]


Sir Thomas Stamford Bingley Raffles, tokoh sentral kolonialisme Britania Raya di Hindia Belanda.
Thomas Stamford Raffles ditunjuk sebagai Letnan Gubernur Jawa oleh pihak Britania Raya.[110] Raffles merombak aturan Belanda yang memberatkan penduduk lokal, seperti Heerendiensten dan perbudakan, tetapi sebagai gantinya menerapkan sistem land tenure (pajak sewa tanah yang dibayarkan oleh penduduk lokal kepada pemerintah kolonial sebagai "tuan tanah") serta menaikkan pajak perorangan. Raffles membentuk pemerintahan yang lebih terpusat dengan tetap mempertahankan para pegawai negeri asal Belanda di tubuh pemerintahannya. Raffles juga berusaha bernegosiasi dengan para penguasa lokal sembari mengurangi hak-hak khusus mereka, serta melancarkan operasi militer kepada penguasa yang membangkang, seperti dalam peristiwa Geger Sepehi di Keraton Yogyakarta.[111][112] Raffles dikenal sebagai peminat sejarah, budaya, dan masyarakat Jawa yang berhasil menyingkap banyak situs kuno yang telah terkubur dan dilupakan pada saat itu, seperti Candi Prambanan (Sleman dan Klaten), Candi Borobudur (Magelang), dan situs-situs Trowulan,[113][114] yang kemudian ditulisnya dalam buku berjudul The History of Java yang terbit pada tahun 1817.[115][116] Selama pemerintahannya, Gunung Tambora di Pulau Sumbawa meletus dengan dahsyat mulai pada tanggal 5 April 1815 dan mencapai puncak erupsi pada tanggal 10–11 April dengan perkiraan skala VEI-7, kemudian berangsur-angsur mereda hingga tanggal 17 April.[117][118] Erupsi ini menyebabkan 71 ribu korban jiwa,[117] serta mungkin menjadi penyebab tahun tanpa musim panas (1816) yang memakan korban belasan ribu jiwa.[119]

Belanda yang keluar dari Kekaisaran Prancis menyetujui suatu perjanjian bersama pihak Britania pada tahun 1814, yang membuat Britania Raya harus mengembalikan koloni milik Belanda sebelum tahun 1803. Perjanjian itu berlaku efektif pada tahun 1815 dan diikuti oleh penurunan jabatan Raffles setahun setelahnya. Koloni di Nusantara sejak tahun 1803 tetap milik Britania Raya, termasuk Bencoolen (Bengkulu), sehingga Raffles dikirim kembali ke Nusantara sebagai Letnan Gubernur Bencoolen pada tahun 1818 dan melakukan eksplorasi ke wilayah Sumatra, Semenanjung Malaya, dan Pulau Ujong (Singapura), meskipun ia dan pasukannya sering kali berseteru dengan pasukan Belanda yang juga ingin memperluas koloninya.[120]

Perluasan wilayah Hindia Belanda
Artikel utama: Hindia Belanda

Johannes van den Bosch, pencetus Cultuurstelsel. Lukisan oleh Raden Saleh.
Pada tanggal 28 Agustus 1814, Belanda membentuk angkatan militer Hindia Belanda yang bernama Tentara Kerajaan Hindia Belanda (KNIL).[121] Setelah lepas dari pengaruh Prancis, Belanda mulai mengklaim koloninya kembali satu per satu dan akhirnya berhasil mengambil alih koloni milik Belanda seperti sedia kala pada tahun 1816. Komisariat Jenderal Hindia Belanda yang dibentuk untuk menata ulang pemerintahan Hindia Belanda kemudian membentuk suatu regeringsreglement (peraturan pemerintah) yang mengatur struktur pemerintahan selama beberapa dekade ke depan serta menyiratkan pandangan politik Pax Nederlandica, yaitu cita-cita Belanda untuk mengolonisasi seluruh Nusantara dan melemahkan kekuasaan penguasa lokal.[122][123] Demi mewujudkan cita-cita tersebut, pemerintah kolonial mulai mengerahkan KNIL ke seluruh kawasan Nusantara demi memperluas wilayah kolonial Hindia Belanda. Pada tahun 1830, Johannes van den Bosch (Gubernur Jenderal Hindia Belanda saat itu) mengakali pengeluaran berlebih yang ditimbulkan oleh ekspedisi KNIL dengan mengeluarkan aturan Cultuurstelsel, yang memaksa pribumi (inlander) menyediakan 20% tanah pertanian untuk tanaman komoditas ekspor Belanda atau bekerja di tanah pertanian milik pemerintah selama 60 hari per tahun.[124] Kebijakan ini terbukti menyelamatkan kas milik pemerintah kolonial, tetapi membuat penduduk lokal semakin sengsara, yang ditambah dengan munculnya bencana kelaparan hebat dan wabah penyakit pada tahun 1840-an.[125] Para penduduk setempat yang merasa sengsara di bawah pemerintahan kolonial mulai melakukan sejumlah pemberontakan dan perlawanan.[126]


Pemisahan wilayah Kesultanan Johor yang menjadi titik awal pembagian wilayah Nusantara menjadi wilayah kolonial Malaya Britania Raya dan Hindia Belanda.
Belanda dan Britania Raya menandatangani perjanjian di London pada tanggal 17 Maret 1824, yang membuat Belanda menyerahkan seluruh koloni di Semenanjung Malaka, Singapura, dan Anak Benua India kepada Britania Raya, sementara Britania Raya menyerahkan koloni di Pulau Sumatra, Riau-Lingga (sekarang Kepulauan Riau), dan Banka-Biliton (sekarang Kepulauan Bangka Belitung) kepada Belanda.[122] Perjanjian tersebut secara praktis membagi wilayah Nusantara menjadi Malaya Britania Raya (sekarang Malaysia dan Singapura) dan Hindia Belanda (sekarang Indonesia).[122] Perjanjian Siak, yang menyetujui pengintegrasian wilayah Siak Sri Inderapura ke dalam Hindia Belanda, disepakati oleh pihak Belanda dan Britania pada tanggal 8 September 1870.[127] Pada tanggal 2 November 1871, Perjanjian Siak diganti dengan Perjanjian Sumatra yang menambahkan seluruh Pulau Sumatra, termasuk Aceh, ke dalam wilayah Hindia Belanda.[128][129]

Pemberontakan oleh rakyat Maluku di bawah komando Pattimura pecah pada bulan Mei 1817 dan berakhir dengan penangkapan dan penjatuhan hukuman gantung terhadap Pattimura dan beberapa tokoh pejuang lainnya.[130]

Pasukan KNIL melakukan penyerangan untuk menguasai Palembang pada tahun 1819 dan dikalahkan oleh yang pasukan pimpinan Mahmud Badaruddin II (Sultan Palembang saat itu), lalu kembali melakukan penyerangan tiba-tiba ke Palembang dua tahun kemudian dan akhirnya berhasil melumpuhkan negara tersebut dan mengasingkan Badaruddin dan keluarganya ke Ternate.[131] Lalu, Hindia Belanda mengirimkan tentara KNIL untuk menaklukkan sisa-sisa pengikut negara bekas Palembang pada tahun 1851–1859.[132] Pada tahun 1864–1868, pasukan KNIL menaklukkan suku Basemah yang meneror Palembang dan Benkoelen (Bengkulu).[133]


Lukisan pertempuran Perang Padri.
Pada tahun 1821, pemerintah kolonial membantu kaum Adat (pendukung tradisi murni Minangkabau) dalam Perang Padri melawan kaum Padri (pendukung syariat Islam) yang terjadi sejak tahun 1803 di Pagaruyung, tetapi akhirnya kalah karena kekurangan pasukan dan menyepakati gencatan senjata dengan kaum Padri pada tahun 1825.[134] Belanda kembali melanjutkan Perang Padri pada tahun 1831, awalnya melawan kaum Padri tetapi kemudian juga melawan kaum Adat yang membelot,[135] hingga akhirnya berhasil memenangkan perang pada tanggal 28 Desember 1838 dengan merebut benteng-benteng kaum Padri dan meruntuhkan negara Pagaruyung.[136] Pada tahun 1841, penduduk Batipuh dan akhirnya beberapa daerah di Pesisir Barat Sumatra melakukan pemberontakan, tetapi berhasil diredam oleh tentara KNIL.[137] Pada tahun 1855–1864, Belanda melancarkan beberapa penyerbuan ke Pulau Nias untuk menaklukkan daerah tersebut.[132]

Di Pulau Kalimantan, pemberontakan di Kalimantan bagian barat pada tahun 1823 pecah karena selisih paham antara pemerintah kolonial dengan orang-orang Tionghoa, tetapi akhirnya berhasil diredam oleh KNIL.[138] Pasukan KNIL kembali menaklukkan pemberontakan orang-orang Tionghoa di Kalimantan yang menolak membayar pajak dan melawan pemerintah kolonial pada tahun 1850–1854.[132] Penduduk Banjar melakukan perlawanan terhadap pasukan KNIL pada tahun 1859–1862 di bawah pimpinan Hidayatullah II, lalu digantikan oleh Antasari.[139]

Pada tahun 1824, Bone membatalkan kerja sama dengan Belanda, sehingga pasukan KNIL dikerahkan untuk menduduki Sulawesi, tetapi kemudian kalah karena kekurangan pasukan, meskipun pemerintah kolonial lalu mengirim pasukan besar beserta artileri pada tahun 1925 untuk melakukan serangan balasan kepada keluarga sultan Bone,[140] hingga akhirnya berhasil menundukkan Bone pada tahun 1838.[141] Pasukan KNIL kembali dikerahkan pada tahun 1859 untuk menumpas pemberontakan Bone.[132]


Lukisan Penangkapan Pangeran Diponegoro, oleh Raden Saleh.
Diponegoro beserta beberapa bangsawan memimpin rakyat Jawa untuk memberontak melawan Belanda dan Yogyakarta sejak tahun 1825,[142] meskipun akhirnya pasukan KNIL berhasil menumpas pasukan Jawa dan memaksa Diponegoro menyerah pada tanggal 28 Maret 1830 dan diasingkan ke Manado, lalu ke Makassar.[142] Pada bagian Pulau Jawa yang lain, para petani Banten yang sengsara akibat bencana dan wabah penyakit memberontak dengan melakukan kerusuhan pada tahun 1888, tetapi dengan cepat diredam oleh pasukan KNIL dalam waktu beberapa hari.

Pertempuran antara KNIL dan para penduduk Bali telah berlangsung beberapa kali melalui perang tahun 1846 di Buleleng, perang tahun 1848 di Buleleng, perang tahun 1849 di Bali utara,[143] pemberontakan tahun 1858 di Buleleng,[144] serta perang dengan orang-orang Sasak pada tahun 1894 di Bali dan Lombok.[145]


Potret foto Teuku Umar, salah satu pahlawan nasional Indonesia.
Belanda melakukan penyerangan ke Aceh pada tahun 1831,[146][147] kemudian melakukan serangkaian penyerangan panjang demi perluasan wilayah selama tahun 1873–1914 di tanah Aceh melawan berbagai pasukan rakyat Aceh yang dipimpin oleh beberapa tokoh pejuang, seperti Mahmud Syah, Muhammad Daud Syah, Teuku Umar, Cut Nyak Dhien, dan Teungku Chik di Tiro.[148][149] Pada wilayah Sumatra lainnya, tentara KNIL dikerahkan untuk menaklukkan tanah Batak dan mendapat perlawanan dari rakyat Batak di bawah komando Sisingamangaraja XII pada tahun 1878–1907.[150] Pada tahun 1885, penduduk Jambi melakukan pemberontakan terhadap Belanda, tetapi berhasil diredam tidak lama kemudian oleh armada kapal KNIL.[151]

Pada tahun 1883, dimulai dengan Puncak Perbuwatan yang mulai mengeluarkan asap pada tanggal 20 Mei, Gunung Krakatau meletus dengan dahsyat selama berbulan-bulan hingga mencapai puncaknya pada tanggal 27 Agustus dan baru dinyatakan selesai pada bulan Oktober.[152] Letusan ini menyebabkan bencana hujan abu vulkanik, gempa bumi, tsunami, dan suara bising yang dahsyat, serta mengakibatkan rusaknya vegetasi di sekitar Selat Sunda dan jatuhnya korban jiwa akibat bencana yang berjumlah sekitar 36 ribu jiwa, serta diperkirakan menjadi penyebab musim dingin vulkanis global dalam kurun waktu empat tahun.[153][154]


Peta ekspansi wilayah kolonial Hindia Belanda.
Memasuki abad ke-20, Belanda berhasil melakukan ekspedisi untuk menguasai wilayah di daerah Kerinci (September 1903) serta menduduki dan membubarkan negara Gowa dan Bone (1905).[155] Belanda juga meredam beberapa pemberontakan, seperti pemberontakan di Pesisir Barat Sumatra akibat belasting (pajak) pada tanggal 15 Juni 1908 yang berhasil diredam dalam waktu sehari oleh korps marsose KNIL.[156]

Pada bulan September 1906, Belanda mengirimkan armada KNIL untuk menduduki kerajaan-kerajaan Bali yang masih bertahan dari pengaruh Belanda.[157] Raja dan para pemangku kerajaan Badung dan Tabanan yang kalah perang melakukan puputan,[158] sementara Dewa Agung Jambe II dari Klungkung awalnya menyerahkan diri dan bersedia menyetujui perjanjian dengan Belanda, tetapi kemudian melakukan pemberontakan bersama pasukannya pada bulan April 1908 yang lalu berhasil ditundukkan oleh armada KNIL.[157][159]

Pada tahun 1920-an, wilayah barat Pulau Papua dimasukkan ke dalam koloni Belanda dan sejak saat itu Hindia Belanda mencakup seluruh wilayah yang saat ini menjadi negara Republik Indonesia.[160]

Pergerakan nasional
Artikel utama: Kebangkitan Nasional Indonesia

Lukisan yang menggambarkan Hindia Belanda sebagai "permata Belanda yang paling berharga". (1916)
Pada tanggal 17 September 1901, Wilhelmina, Ratu Belanda pada saat itu, mengemukakan Politik Etis, yang menyebutkan bahwa Kerajaan Belanda memiliki utang budi (eerschuld) terhadap kaum pribumi Hindia Belanda, dan oleh karenanya mengumumkan program kebijakan Trias van Deventer demi membalas budi kaum pribumi, salah satunya adalah meningkatkan taraf pendidikan pribumi dengan membuka sekolah-sekolah. Kebijakan tersebut memberi sumbangsih terhadap kemunculan gerakan-gerakan kebangsa di tanah Hindia Belanda.[161]

Pada tanggal 20 Mei 1908, beberapa pelajar dari School tot Opleiding van Inlandsche Artsen (STOVIA) mendirikan Boedi Oetomo, yang menjadi pelopor gerakan kebangkitan nasional Indonesia di Hindia Belanda.[162][163] Pada akhirnya, Boedi Oetomo bergabung dengan beberapa kelompok kedaerahan lain dan membentuk Partij Indonesia Raja.[164]

Pada tanggal 5 April 1909, serikat dagang bernama Sarekat Dagang Islam mulai beroperasi dengan membuka cabang di Batavia (sekarang Jakarta) dan Buitenzorg (sekarang Bogor).[165] Namanya diubah pada tahun 1912 oleh Oemar Said Tjokroaminoto menjadi Sarekat Islam (SI).[166] Pada bulan Oktober 1921, SI menyatakan bahwa anggotanya tidak boleh merangkap keanggotaan pada organisasi lain, menyebabkan anggota-anggota yang menolak melepaskan keanggotaan lainnya terpaksa keluar dari organisasi.[167] Pada tahun 1929, namanya diubah lagi menjadi Partai Sarekat Islam Indonesia.[168]


Potret Tiga Serangkai ketika di pengasingan (1914).
Pada tanggal 25 Desember 1912, "Tiga Serangkai" yang terdiri dari Ernest Douwes Dekker, Tjipto Mangoenkoesoemo, dan Soewardi Soerjaningrat (yang mengganti namanya menjadi Ki Hadjar Dewantara di kemudian hari)[169] mendirikan Indische Partij dengan tujuan memperjuangkan hak-hak kaum Indo dan pribumi melalui jalur politik, meskipun akhirnya dibubarkan oleh Pemerintah Hindia Belanda.[170][171] Setelahnya, Tiga Serangkai masih tetap memperjuangkan pandangan mereka dengan menulis kritik-kritik kepada pemerintah kolonial melalui koran-koran, tetapi akibatnya mereka ditangkap dan diasingkan ke Belanda oleh pemerintah.[172]

Selain perjuangan politik, beberapa tokoh juga mendirikan sekolah-sekolah demi mencerdaskan kehidupan bangsa dan memperjuangkan emansipasi kaum wanita. Sekembalinya dari pengasingan, Ki Hadjar Dewantara mendirikan lembaga pendidikan Taman Siswa, yang dimulai pada tanggal 3 Juli 1922 di Yogyakarta dan lalu menyebar ke seluruh Jawa dan bahkan ke luar pulau.[173][174] Sakola Istri (kemudian berganti menjadi Sakola Kaoetamaan Isteri) didirikan oleh seorang wanita priayi Sunda bernama Dewi Sartika pada tahun 1904 demi mencerdaskan kehidupan wanita di Tanah Sunda.[175] Kartinischool yang dibuka sejak tahun 1912 oleh Conrad van Deventer dan Elisabeth Maas terinspirasi dari surat-surat yang ditulis oleh Kartini, seorang wanita priayi Jawa yang kritis akan masalah-masalah sosial di sekitarnya, termasuk masalah ketimpangan gender.[176]

Pada tanggal 23 Mei 1914, Henk Sneevliet membentuk serikat pekerja bernama Indische Sociaal Democratische Vereeniging dengan tujuan menyebarkan paham komunisme dan menentang pemerintah kolonial.[177] Pada bulan Mei 1920, ISDV berganti nama menjadi Persarekatan Kommunist India, kemudian mengganti namanya lagi menjadi Partij Kommunist Indonesia (PKI) pada tahun 1924.[178] Paham komunisme kemudian menyebar ke organisasi-organisasi lain, termasuk Sarekat Islam.[179] PKI mengadakan pemberontakan terhadap pemerintah kolonial, yang dimulai di Labuan pada tanggal 12 November 1926 dengan menyerang para pegawai pemerintah di kediaman masing-masing. Pemberontakan meluas di wilayah Jawa dan Sumatra, hingga akhirnya berhasil diredam seluruhnya oleh pasukan militer KNIL pada tanggal 28 Februari 1927.[180][181] Sejak itu, PKI ditetapkan sebagai organisasi terlarang di Hindia Belanda.


Mohammad Hatta, tokoh pemimpin Perhimpunan Indonesia yang kelak menjadi Wakil Presiden Indonesia pertama.
Pada tahun 1908, Indische Vereeniging dibentuk oleh Soetan Kasajangan Soripada dan Noto Soeroto sebagai wadah pemersatu para pelajar Hindia di perantauan Belanda, yang kemudian menjadi wadah pelajar tersebut untuk menyuarakan pandangan politik.[182] Pada bulan September 1922, perkumpulan ini mengganti namanya menjadi Indonesische Vereeniging, menjadikannya sebagai organisasi pertama yang menggunakan nama "Indonesia". Organisasi ini kemudian berganti nama lagi menjadi Perhimpoenan Indonesia (PI) dengan menggunakan versi bahasa Melayu sebagai nama resmi.[183] Selama Mohammad Hatta menjabat sebagai Ketua PI, beliau beserta beberapa tokoh lainnya pernah ditangkap oleh Pemerintah Belanda karena dituduh berkomplot dengan PKI, meskipun akhirnya dibebaskan karena kurangnya bukti.[184][185]

Pada tanggal 4 Juli 1927, Soekarno, Tjipto Mangoenkoesoemo, Sartono, dan Iskaq Tjokrohadisoerjo, mendirikan Persarekatan National Indonesia, yang kemudian berganti nama menjadi Partij National Indonesia (PNI) pada bulan Mei 1928, dengan tujuan memperjuangkan kemerdekaan atas wilayah Hindia Belanda tanpa kerja sama rezim kolonial Belanda.[186] Kepopuleran Soekarno dan PNI membuat pemerintah kolonial merasa terancam, sehingga Soekarno dan beberapa petinggi partai ditangkap pada bulan Desember 1929,[187][188] lalu dijatuhi pidana penjara pada tanggal 22 Desember 1930 atas dalih persekongkolan untuk menggulingkan pemerintah kolonial.[189] PNI lalu terpecah menjadi organisasi Pendidikan National Indonesia (PNI "Baru") dan Partij Indonesia (Partindo). Soekarno yang dibebaskan lebih awal pada tanggal 31 Desember 1931 kemudian memilih masuk ke Partindo dan kemudian menjadi ketua organisasi tersebut pada tanggal 28 Juli 1932.[190] Hatta yang pulang dari Belanda pada bulan Juli menjadi anggota PNI Baru dan akhirnya diangkat sebagai ketua pada bulan Agustus 1932.[191] Oleh karena tulisan-tulisan mereka yang mendukung dan mendorong kemerdekaan, Soekarno kembali ditangkap dan ditahan pada bulan Agustus 1933, diikuti oleh Hatta dan Sjahrir pada awal tahun 1934, lalu masing-masing diasingkan ke beberapa tempat yang berbeda.[192]


Museum Sumpah Pemuda (dahulu Indonesische Clubhuis) merupakan lokasi rapat terakhir Kongres Pemuda II sekaligus menjadi tempat lahirnya Sumpah Pemuda.
Beberapa gerakan kepemudaan mengadakan Kongres Pemuda I yang dipimpin oleh Mohammad Tabrani dan terdiri dari tiga pertemuan pada tanggal 30 April sampai 2 Mei 1926 di Vrijmetselaarsloge (sekarang Gedung Badan Perencanaan Pembangunan Nasional). Dalam rapat terakhir, Mohammad Jamin dari Jong Sumatranenbond mengusulkan bahasa Melayu menjadi bahasa persatuan, kemudian Mohammad Tabrani Soerjowitjirto dari Jong Java mengusulkan penggunaan istilah "bahasa Indonesia" alih-alih bahasa Melayu.[193] Dua tahun kemudian, Kongres Pemuda II yang terdiri dari tiga rapat dan diketuai oleh Soegondo Djojopoespito diadakan pada tanggal 27–28 Oktober dan diikuti oleh beberapa perhimpunan kepemudaan, gerakan nasionalisme dan agama, serta kelompok belajar dari sekolah-sekolah tertentu.[194] Di sela-sela rapat terakhir, lagu "Indonesia Raja" diperdengarkan di hadapan hadirin, awalnya berupa instrumental biola oleh sang penggubah Wage Rudolf Soepratman dan diulang dengan iringan nyanyian Dolly Salim, putri sulung Agoes Salim.[195] Pada akhir kongres, suatu naskah resolusi yang dibuat oleh Jamin dibacakan oleh Soegondo di depan hadirin dan menjadi ikrar bagi seluruh peserta kongres. Ikrar tersebut kini dikenal sebagai Sumpah Pemuda.[196]

Menyadari maraknya gerakan-gerakan nasionalisme yang menuntut kemerdekaan dari kolonialisme Belanda, pemerintah kolonial mulai melarang dan menutup beberapa organisasi serta menangkap sejumlah tokoh pergerakan.[197]

Periode pendudukan
Pendudukan Jepang
Artikel utama: Sejarah Nusantara (1942–1945)

Kapal HMS Exeter yang tenggelam pada saat Pertempuran Laut Jawa II.
Pecahnya Perang Dunia II benar-benar melemahkan kekuatan pertahanan Belanda, terutama ketika Belanda jatuh ke tangan Jerman Nazi pada tanggal 14 Mei 1940.[198] Pada bulan Januari 1942, Jepang menggunakan kesempatan tersebut dengan mulai memasuki teritori Hindia Belanda dan melancarkan operasi militer melawan pasukan gabungan ABDACOM (Komando Amerika Serikat–Britania Raya–Belanda–Australia) yang menjaga kawasan tersebut, terutama di area Kalimantan bagian barat, Tarakan, Manado, Balikpapan, Kendari, Samarinda, Banjarmasin, Ambon, Selat Makassar, Pulau Sumatra (terutama Palembang), Selat Badung, Pulau Timor, Laut Jawa, Selat Sunda, serta beberapa titik di Pulau Jawa, seperti di Kalijati, Leuwiliang, dan Ciater.[199] Strategi militer Jepang berhasil memojokkan lawannya, hingga seluruh pasukan ABDACOM berhasil dilumpuhkan pada Pertempuran Laut Jawa terakhir tanggal 1 Maret.[200] Setelah memegang kendali, Jepang memecah Hindia Belanda menjadi daerah Sumatra (dan awalnya Malaya) di bawah komando Angkatan Darat XXV yang berpusat di Bukittinggi (sebelumnya di Singapura), daerah Jawa dan Madura di bawah komando Angkatan Darat XVI yang berpusat di Batavia (kemudian menjadi Jakarta), serta wilayah Hindia lainnya di bawah Angkatan Laut Kekaisaran Jepang yang berpusat di Makassar.[201]


Poster propaganda Gerakan 3A.
Berbekal propaganda Gerakan 3A, Jepang awalnya diterima sangat baik oleh sebagian besar penduduk lokal Hindia Belanda yang mengalami kekerasan oleh pemerintah kolonial.[202] Namun kenyataannya, Jepang merekrut paksa banyak penduduk lokal untuk dijadikan pekerja paksa (rōmusha) dan wanita penghibur (ianfu) untuk menyokong pasukan mereka yang sedang berperang melawan Blok Sekutu, menyebabkan banyaknya penderitaan dan kematian akibat kekerasan, kecelakaan kerja, bencana kelaparan, dan penyakit kelamin.[203][204] Beberapa pemberontakan melawan pendudukan Jepang dilancarkan oleh penduduk setempat, tetapi dapat diredam dengan cepat oleh pasukan Jepang,[205] misalnya dua pemberontakan Tjirebon (Cirebon) dan satu pemberontakan Sukamanah yang dapat diredam oleh pasukan Jepang, insiden Peristiwa Mandor dan rangkaian Perang Dayak Desa,[206][207] serta pemberontakan rakyat Aceh melawan pasukan Jepang dan Sekutu sekaligus.[208]

Sejak tahun 1944, gempuran yang tiada habisnya dari Blok Sekutu makin melemahkan pertahanan pasukan Jepang. Pasukan gabungan Belanda dan Britania Raya, yang dibantu oleh Amerika Serikat dan Australia, mulai melancarkan serangkaian operasi militer untuk merebut kembali wilayah koloni mereka dari Jepang di Pulau Papua bagian barat sejak tanggal 22 April 1944 dan Pulau Kalimantan (terutama di Tarakan dan Balikpapan) sejak tanggal 1 Mei 1945.[209][210] Rangkaian operasi tersebut berakhir ketika Jepang menyerah dan Perang Pasifik berakhir.

Persiapan kemerdekaan
Artikel utama: Proklamasi Kemerdekaan Indonesia

Suasana sidang BPUPK yang kedua pada tanggal 10–17 Juli 1945.
Demi menggalang dukungan dari penduduk lokal, Jepang mulai menjanjikan kemerdekaan kepada tokoh-tokoh pergerakan. Pemerintah militer AD XVI mengumumkan pembentukan Badan Penyelidik Usaha-Usaha Persiapan Kemerdekaan (BPUPK) pada tanggal 1 Maret 1945 dan diresmikan pada tanggal 29 April dengan anggota berjumlah 67 orang dan ketua Radjiman Wedyodiningrat.[211] Sidang pertama pada tanggal 28 Mei hingga 1 Juni di Gedung Tyuuoo Sangi-in (sekarang Gedung Pancasila) mendiskusikan tentang ideologi negara. Pada hari terakhir, Soekarno berpidato mengenai lima dasar negara merdeka yang dinamakan "Pancasila".[212] Oleh karena sidang ini belum memberikan kata mufakat, Panitia Kecil dibentuk untuk merumuskan dasar negara, tetapi dirombak menjadi Panitia Sembilan pada tanggal 18 Juni,[213] dan akhirnya menghasilkan rumusan yang bernama Piagam Jakarta pada tanggal 22 Juni.[214] Sidang kedua pada tanggal 10–17 Juli membahas segala permasalahan mendasar suatu negara, hingga mencapai kata mufakat akan bentuk negara kesatuan republik, wilayah Indonesia Raya, dan Undang-Undang Dasar 1945 (UUD 1945) sebagai konstitusi negara.[215] Di Sumatra, BPUPK serupa juga dibentuk oleh pemerintah AD XXV pada tanggal 25 Juli, tetapi tidak pernah sempat mengadakan sidang.[216]


Pembacaan deklarasi kemerdekaan oleh Soekarno.

Deklarasi kemerdekaan Indonesia
Duration: 50 detik.0:50
Reka ulang oleh Soekarno pada tahun 1950 atau 1951.
Bermasalah memainkan berkas ini? Lihat bantuan media.
Jepang mengumumkan pembentukan Panitia Persiapan Kemerdekaan Indonesia (PPKI) pada tanggal 7 Agustus.[217] Jepang menyerah kepada Blok Sekutu pada tanggal 15 Agustus, tetapi berita itu tidak diumumkan secara resmi di Hindia Belanda, meskipun akhirnya diketahui oleh para pemuda Menteng 31 melalui siaran BBC Radio, yang kemudian berpendapat bahwa kemerdekaan harus dideklarasikan sesegera mungkin dan tanpa melibatkan PPKI yang dianggap sebagai "boneka" Jepang.[218] Pada tanggal 16 Agustus dini hari, Soekarno dan Hatta yang tidak setuju akan pendapat itu diculik dan dibawa ke Rengasdengklok.[219] Setelah perundingan yang alot, Soekarno dan Hatta yang akhirnya setuju lalu kembali ke Jakarta, kemudian berkumpul bersama beberapa tokoh lain pada besok dini hari di kediaman Tadashi Maeda (sekarang Museum Perumusan Naskah Proklamasi) untuk merumuskan deklarasi kemerdekaan. Naskah deklarasi ditik oleh Sajuti Melik sesuai rancangan yang telah disepakati, serta ditandatangani oleh Soekarno dan Hatta.[220] Proklamasi Kemerdekaan Indonesia dilaksanakan di kediaman Soekarno (sekarang menjadi Taman Proklamasi) pada tanggal 17 Agustus 2605 tahun Jepang, sekitar pukul 10.00 waktu Jepang.[b][221] Setelah pidato proklamasi Soekarno, upacara pengibaran Sang Merah Putih pertama (kelak menjadi Bendera Pusaka) dilaksanakan di halaman rumah dan diiringi nyanyian Indonesia Raya oleh hadirin.[222]

Periode republik
Revolusi nasional
Artikel utama: Sejarah Indonesia (1945–1949) dan Revolusi Nasional Indonesia

Suasana sidang pertama Panitia Persiapan Kemerdekaan Indonesia pada tanggal 18 Agustus 1945.
Sehari setelah proklamasi, PPKI mengadakan sidang pertama di gedung bekas Dewan Hindia (sekarang Gedung BP7) dan memutuskan pengesahan UUD 1945 serta pengangkatan Soekarno dan Hatta sebagai pasangan Presiden dan Wakil Presiden.[223] Sidang kedua keesokan harinya menghasilkan pembentukan provinsi dan pengangkatan menteri.[224] Pada tanggal 22 Agustus 1945, PPKI mengadakan sidang terakhir dan memutuskan pembentukan Komite Nasional Indonesia Pusat (KNIP), Partai Nasional Indonesia (PNI) dan Badan Keamanan Rakyat (BKR).[225] PPKI dibubarkan pada tanggal 29 Agustus.[226]

Selama sebulan, berita mengenai deklarasi kemerdekaan Indonesia tersebar luas hingga ke luar Pulau Jawa, sehingga banyak penduduk "pro Republik" yang mulai menyerang orang-orang asing dan orang-orang "pro Belanda" demi merebut segala aset mereka.[227] Pada bulan September 1945, militer Britania Raya datang terlebih dahulu ke Indonesia, terutama ke Jawa, untuk mengamankan wilayah dengan dibantu oleh beberapa pasukan Jepang. Perang antara militer Britania melawan pasukan pro Republik pun pecah di berbagai tempat, seperti dalam peristiwa Penyerbuan Kotabaru, Pertempuran Bojongkokosan, Pertempuran Lima Hari di Semarang, dan Palagan Ambarawa.[228] Sementara itu, militer Belanda di bawah Pemerintahan Sipil Hindia Belanda (NICA) secara bertahap masuk untuk menempati kembali pos-pos Belanda di Indonesia. Pada tanggal 10 November (sekarang menjadi Hari Pahlawan), tentara Britania Raya memulai Pertempuran Surabaya yang berlangsung selama bebeberapa hari dan menyebabkan belasan ribu korban jiwa berjatuhan.[229] Pertempuran tersebut menyadarkan pihak Britania Raya dan Belanda bahwa pemerintahan Republik didukung secara luas oleh rakyat setempat.[230] Britania Raya mulai mengambil sikap netral setelahnya, tetapi NICA dan para pendukung Belanda tetap melancarkan serangan terhadap pasukan pro republik pada akhir tahun 1945, seperti pada Pertempuran Medan Area, Peristiwa 19 November di Kolaka, Perang Cumbok di Pidie, Perang Dayak Desa, dan Pertempuran Kumai.


Pembagian wilayah yang disepakati dalam Perjanjian Linggarjati.
Akibat situasi Jakarta yang semakin tidak kondusif, Soekarno dan Hatta beserta beberapa menteri dan staf, dengan keluarga masing-masing, pindah ke Yogyakarta pada tanggal 3–4 Januari 1946 untuk mendirikan ibu kota di sana,[231] tetapi Sutan Sjahrir (Perdana Menteri pada saat itu) dan beberapa petinggi tetap tinggal di Jakarta untuk mengadakan negosiasi dengan pihak Belanda.[232] Sepanjang tahun 1946, para pendukung Republik tetap melakukan berbagai penyerangan dan pertahanan melawan militer Belanda dan pasukan pendukungnya, terutama di luar Pulau Jawa yang memiliki sentimen dukungan yang lebih rendah, seperti dalam peristiwa Pertempuran Lengkong, Bandung Lautan Api, Revolusi Sosial di Sumatera Timur, dan Pertempuran Selat Bali.[233] Pada tanggal 15 November 1946 di kediaman keluarga Kwee (sekarang Gedung Perundingan Linggarjati), pihak Republik (dipimpin oleh Sjahrir) dan pihak Belanda (dipimpin oleh Willem Schermerhorn) menyepakati Perjanjian Linggarjati yang mengakui kekuasaan de facto Republik Indonesia atas Sumatra, Jawa, dan Madura sebagai negara konstituen di dalam negara federasi yang akan dibentuk, yang kemudian diratifikasi oleh Belanda pada tanggal 19 Desember dan oleh Indonesia pada tanggal 25 Maret 1947.[234][235] Ketidakpuasan yang muncul akibat perjanjian ini menyebabkan beberapa insiden pun terjadi, seperti dalam peristiwa Puputan Margarana, Pembantaian Westerling di Sulawesi Selatan, Peristiwa Tiga Maret di Sumatera Barat, Pertempuran Lima Hari Lima Malam di Palembang, Pertempuran Laut Cirebon, dan Pertempuran Laut Sibolga.


Peta kekuasaan Republik Indonesia (merah) dan Belanda (krem) di Pulau Jawa menurut Perjanjian Renville.
Pada tanggal 21 Juli hingga 5 Agustus 1947, Belanda melanggar Perjanjian Linggarjati dengan melancarkan Operatie Product (Operasi Produk) atau Agresi Militer Belanda I demi merebut area-area produktif yang dikuasai oleh Republik, tetapi berdalih sedang melakukan politionele actie ("aksi polisionil") demi memulihkan keamanan dan ketertiban yang kacau, seperti pada Tragedi Mergosono.[236] Belanda nyatanya berhasil merebut sebagian besar daerah tersebut, tetapi operasi tersebut mendapat kecaman dari dunia internasional.[237] Resolusi 27 Dewan Keamanan Perserikatan Bangsa-Bangsa (DK PBB) yang menyerukan gencatan senjata bagi pihak Belanda dan pihak Indonesia menyebabkan Belanda terpaksa menyanggupinya.[238] Mulai pada tanggal 8 Desember di geladak kapal USS Renville, pihak Indonesia (dipimpin oleh Amir Sjarifoeddin) dan pihak Belanda (dipimpin oleh Abdulkadir Widjojoatmodjo) melakukan perundingan ulang. Meskipun beberapa insiden terjadi selama perundingan (seperti pada Pembantaian Rawagede), kedua pihak berhasil menyepakati dan meratifikasi Perjanjian Renville pada tanggal 17 Januari 1948, yang berisi pengakuan atas garis demarkasi yang dijuluki Status Quo Lijn (Garis Status Quo) atau "Garis van Mook", yang membagi sepertiga wilayah Jawa dan sebagian besar Sumatra yang dikuasai oleh Republik Indonesia dengan wilayah lain yang dikuasai oleh Belanda.[239][240]


Lukisan Soedirman yang sedang ditandu sembari memimpin pasukan gerilya. Dilukis oleh Hardjanto.
Para tokoh oposisi Front Demokrasi Rakyat (FDR) mulai melakukan pemberontakan di Madiun pada tanggal 18 September 1948 dan sebagian besar dari mereka berhasil ditangkap dan dieksekusi dalam waktu tiga bulan oleh Tentara Nasional Indonesia (TNI).[241] Namun, kejadian tersebut menjadi peluang bagi Belanda untuk melancarkan Operatie Kraai (Operasi Gagak) atau Agresi Militer Belanda II pada tanggal 19 Desember demi membubarkan Republik.[242] Dalam waktu singkat, pasukan Belanda berhasil menguasai Yogyakarta serta membuang Soekarno, Hatta, dan beberapa petinggi lain ke Bangka, sementara pasukan TNI yang tersisa melancarkan pertempuran gerilya di bawah komando Soedirman selama beberapa bulan. Syafruddin Prawiranegara yang mendengar berita tersebut ketika di Bukittinggi segera membentuk Pemerintahan Darurat Republik Indonesia (PDRI),[243] sementara dunia internasional sekali lagi mengecam tindakan Belanda, hingga DK PBB mengeluarkan Resolusi 63 yang menyerukan penghentian pertikaian dan menuntut pembebasan tawanan perang.[244] Demi menampik tuduhan bahwa Indonesia telah bubar, pasukan TNI melancarkan serangan umum terhadap pasukan Belanda di Yogyakarta pada tanggal 1 Maret 1949 dan nyatanya berhasil menguasai kota tersebut selama enam jam.[245] Pada tanggal 7 Mei, pihak Indonesia (dipimpin oleh Mohamad Roem) dan pihak Belanda (dipimpin oleh Jan Herman van Roijen) menyepakati Perjanjian Roem–van Roijen, yang pada intinya menyetujui pengembalian status quo sebelum Agresi II dan kesediaan Indonesia untuk ikut dalam konferensi pembentukan federasi.[246] Pasukan TNI kembali melakukan serangan umum terhadap tentara Belanda di Surakarta pada tanggal 7 Agustus dan berhasil menduduki kota tersebut hingga tanggal 10 Agustus.[247]


Sesi terakhir Konferensi Meja Bundar (KMB) di Ridderzaal, Den Haag pada tanggal 2 November 1949.
Para petinggi yang telah dilepaskan dari pengasingan tiba di Yogyakarta pada tanggal 6 Juli, kemudian menyetujui Perjanjian Roem–van Roijen dan sekaligus mengambil kembali mandat pemerintahan dari PDRI pada tanggal 13 Juli. Sekarmadji Maridjan Kartosoewirjo yang sejak awal menolak Republik Indonesia akhirnya mendirikan Negara Islam Indonesia (NII) dan membentuk Tentara Islam Indonesia (TII) pada tanggal 7 Agustus.[248] Pada tanggal 23 Agustus, pihak Belanda, pihak Republik, dan utusan Majelis Permusyawaratan Federal (BFO, beranggotakan negara-negara boneka bentukan Belanda) memulai Konferensi Meja Bundar (KMB) di Den Haag, Belanda. Pada tanggal 2 November, semua pihak menandatangani Perjanjian Meja Bundar, yang menyetujui pembentukan negara Republik Indonesia Serikat (RIS) dan konstitusinya serta menjanjikan penyerahan kedaulatan tanpa syarat dari Belanda kepada RIS.[249] Perjanjian tersebut diratifikasi oleh KNIP pada tanggal 14 Desember dan Dewan Negara Belanda pada tanggal 21 Desember.[250] Akhirnya pada tanggal 27 Desember, Juliana (Ratu Belanda) dari pihak Belanda bersama Mohammad Hatta dari pihak Indonesia menandatangani Akta Penyerahan Kedaulatan (Akte van Soevereiniteitsoverdracht) di Istana Kerajaan Amsterdam, Belanda.[251] Upacara serupa diadakan pada hari yang sama di Istana Rijswijk (sekarang Istana Negara), dengan Antonius Lovink (Komisaris Tinggi Hindia Belanda) dari pihak Belanda dan Hamengkubuwana IX dari pihak Indonesia.[252]

Demokrasi federal
Artikel utama: Republik Indonesia Serikat

Pembagian administratif di Republik Indonesia Serikat (RIS).
Sebagai akibat dari Perjanjian Meja Bundar, Republik Indonesia Serikat (RIS) terbentuk secara resmi pada tanggal 27 Desember 1949, dengan Republik Indonesia (RI) sebagai salah satu negara bagian RIS. Namun akibat anggapan bahwa konsep negara federal merupakan bentuk lain dari kolonialisme dan kurang mumpuni dalam menyatukan rakyat Indonesia yang sangat majemuk, negara RIS tidak dapat bertahan lama.[253] Pada tanggal 22–23 Januari 1950, Angkatan Perang Ratu Adil (APRA) di bawah pimpinan Raymond Westerling menyerang pasukan Tentara Nasional Indonesia (TNI) di Bandung dan Jakarta demi menggulingkan Pemerintahan RIS, meskipun akhirnya gagal.[254] Di saat yang sama, Tentara Islam Indonesia (TII) di bawah kepemimpinan Amir Fatah juga melancarkan aksi pemberontakan di Tegal Raya.[255] Pada bulan Maret–April 1950, hampir seluruh entitas konstituen dalam RIS membubarkan diri secara sukarela, hingga tersisa RI, Sumatera Timur, dan Indonesia Timur.[256] Pada tanggal 19 Mei, Pemerintah RIS mengumumkan piagam persetujuan yang berisi kesepakatan bersama ketiga negara bagian untuk "membentuk negara kesatuan sebagai penjelmaan dari negara Republik Indonesia yang berdasarkan pada Proklamasi 17 Agustus 1945".[257][258] Setelah dilakukan sejumlah persiapan, Soekarno secara resmi membubarkan RIS dan melanjutkan entitas Negara Kesatuan Republik Indonesia (NKRI).[259]

Demokrasi liberal
Artikel utama: Sejarah Indonesia (1950–1959)

Soekarno, presiden pertama Indonesia.

Piagam Penyerahan Kedaulatan Indonesia oleh Belanda
Pada tahun 1950-an dan 1960-an, pemerintahan Soekarno mulai mengikuti sekaligus merintis gerakan non-blok pada awalnya, kemudian menjadi lebih dekat dengan blok sosialis, misalnya Republik Rakyat Tiongkok dan Yugoslavia.

Demokrasi terpimpin
Artikel utama: Sejarah Indonesia (1950–1959)
Tahun 1960-an menjadi saksi terjadinya konfrontasi militer terhadap negara tetangga, Malaysia ("Konfrontasi"),[260] dan ketidakpuasan terhadap kesulitan ekonomi yang semakin besar. Selanjutnya pada tahun 1965 meletus peristiwa G30S yang menyebabkan kematian 6 orang jenderal dan sejumlah perwira menengah lainnya. Muncul kekuatan baru yang menyebut dirinya Orde Baru yang segera menuduh Partai Komunis Indonesia sebagai otak di belakang kejadian ini dan bermaksud menggulingkan pemerintahan yang sah serta mengganti ideologi nasional menjadi berdasarkan paham sosialis-komunis. Tuduhan ini sekaligus dijadikan alasan untuk menggantikan pemerintahan lama di bawah Presiden Soekarno.

Transisi
Artikel utama: Sejarah Indonesia (1965–1966)
Orde baru
Artikel utama: Orde Baru

Potret resmi Soeharto, Presiden Indonesia ke-2, pada tahun 1993.
Jenderal Soeharto menjadi Pejabat Presiden pada tahun 1967 dengan alasan untuk mengamankan negara dari ancaman komunisme. Sementara itu kondisi fisik Soekarno sendiri semakin melemah. Setelah Soeharto berkuasa, ratusan ribu warga Indonesia yang dicurigai terlibat pihak komunis dibunuh, sementara masih banyak lagi warga Indonesia yang sedang berada di luar negeri, tidak berani kembali ke tanah air, dan akhirnya dicabut kewarganegaraannya. Tiga puluh dua tahun masa kekuasaan Soeharto dinamakan Orde Baru, sementara masa pemerintahan Soekarno disebut Orde Lama.

Soeharto menerapkan ekonomi neoliberal dan berhasil mendatangkan investasi luar negeri yang besar untuk masuk ke Indonesia dan menghasilkan pertumbuhan ekonomi yang besar, meski tidak merata. Pada awal rezim Orde Baru kebijakan ekonomi Indonesia disusun oleh sekelompok ekonom lulusan Departemen Ekonomi Universitas California, Berkeley, yang dipanggil "Mafia Berkeley".[261] Namun, Soeharto menambah kekayaannya dan keluarganya melalui praktik korupsi, kolusi, dan nepotisme yang meluas dan dia akhirnya dipaksa turun dari jabatannya setelah aksi demonstrasi besar-besaran dan kondisi ekonomi negara yang memburuk pada tahun 1998.

Reformasi
Artikel utama: Sejarah Indonesia (1998–sekarang)
Masa Peralihan Orde Reformasi atau Era Reformasi berlangsung dari tahun 1998 hingga 2001, ketika terdapat tiga masa presiden: Bacharuddin Jusuf (BJ) Habibie, Abdurrahman Wahid dan Megawati Soekarnoputri. Pada tahun 2004, diselenggarakan Pemilihan Umum satu hari terbesar di dunia[262] yang dimenangkan oleh Susilo Bambang Yudhoyono, sebagai presiden terpilih secara langsung oleh rakyat, yang menjabat selama dua periode. Pada tahun 2014, Joko Widodo, yang lebih akrab disapa Jokowi, terpilih sebagai presiden ke-7.

Indonesia kini sedang mengalami masalah-masalah ekonomi, politik dan pertikaian bernuansa agama di dalam negeri, dan beberapa daerah berusaha untuk melepaskan diri dari naungan NKRI, terutama Papua.[butuh rujukan] Timor Timur secara resmi memisahkan diri pada tahun 1999 setelah 24 tahun bersatu dengan Indonesia dan 3 tahun di bawah administrasi PBB menjadi negara Timor Leste.

Pada Desember 2004 dan Maret 2005, Aceh dan Nias dilanda dua gempa bumi besar yang totalnya menewaskan ratusan ribu jiwa. (Lihat Gempa bumi Samudra Hindia 2004 dan Gempa bumi Sumatra Maret 2005.) Kejadian ini disusul oleh gempa bumi di Yogyakarta dan tsunami yang menghantam Pantai Pangandaran dan sekitarnya, serta banjir lumpur di Sidoarjo pada 2006 yang tidak kunjung terpecahkan.

Geografi
Artikel utama: Geografi Indonesia
Lihat pula: Asia § Peta, dan Daftar pulau di Indonesia menurut provinsi

Hutan hujan di Taman Nasional Gunung Palung, Kalimantan Barat.
Indonesia merupakan negara kepulauan terbesar di dunia yang berada di Asia Tenggara,[263] dan terletak di antara benua Asia dan benua Australia/Oseania, serta di antara Samudra Hindia dan Samudra Pasifik. Negara ini memiliki 17.504 pulau yang menyebar di sekitar khatulistiwa; sebanyak 16.056 pulau telah dibakukan namanya,[264] dan sekitar 6.000 pulau tidak berpenghuni.[265][266] Pulau-pulau besar di Indonesia yaitu Sumatra, Jawa, Kalimantan (berbagi dengan Malaysia dan Brunei Darussalam), Sulawesi, dan Papua (berbagi dengan Papua Nugini).

Indonesia berada pada koordinat antara antara 6° 04' 30" LU dan 11° 00' 36" LS serta antar 94° 58' 21" dan 141° 01' 10" BT,[267] yang membentang sepanjang 5.120 kilometer (3.181 mil) dari timur ke barat serta 1.760 kilometer (1.094 mil) dari utara ke selatan.[268] Luas daratan Indonesia adalah 1.916.906,77 km²,[269] sementara luas perairannya sekitar 3.110.000 km² dengan garis pantai sepanjang 108 ribu km.[270] Batas wilayah Indonesia diukur dari kepulauan dengan menggunakan teritorial laut 12 mil laut serta zona ekonomi eksklusif 200 mil laut,[271] searah penjuru mata angin, yaitu:

Utara	Malaysia dengan perbatasan sepanjang 1.782 km,[265] Singapura, Filipina, dan Laut Tiongkok Selatan
Timur	Papua Nugini dengan perbatasan sepanjang 820 km,[265] Timor Leste, dan Samudra Pasifik
Selatan	Australia, Timor Leste, dan Samudra Hindia
Barat	Samudra Hindia
Titik tertinggi di Indonesia yaitu Puncak Jaya (4.884 mdpl) di Provinsi Papua Tengah.[272] Danau Toba di Sumatera Utara adalah danau terluas di Indonesia sekaligus danau kaldera terbesar di dunia,[273] sedangkan sungai terpanjang di Indonesia yaitu Sungai Kapuas yang berada di Kalimantan Barat.[274]

Iklim
Artikel utama: Iklim Indonesia dan Perubahan iklim di Indonesia

Peta klasifikasi Iklim Köppen Indonesia.
Secara umum, Indonesia beriklim tropis (kelompok A dalam klasifikasi iklim Köppen; meskipun ada wilayah dengan tipe iklim yang berbeda).[275][276] Perairan yang hangat di wilayah Indonesia sangat berperan dalam menjaga suhu di darat tetap konstan, dengan rerata suhu di wilayah pesisir sebesar 28 °C, di wilayah pedalaman dan dataran tinggi sebesar 26 °C , serta di wilayah pegunungan sebesar 23 °C. Kelembapan berkisar antara 70 hingga 90%.[277]

Faktor utama yang memengaruhi iklim Indonesia bukanlah suhu udara ataupun tekanan udara, melainkan curah hujan. Variasi musim di Indonesia, yaitu musim hujan dan musim kemarau, berkaitan dengan pergerakan angin muson. Angin muson barat yang bertiup dari Asia ke Australia melalui Indonesia pada bulan Oktober hingga Februari mengakibatkan curah hujan yang tinggi, terutama di Indonesia bagian barat. Sementara itu, angin muson timur yang bergerak ke arah sebaliknya pada bulan April hingga Agustus tidak banyak membawa uap air dan menurunkan hujan. Selain itu, ada pula musim peralihan ketika matahari melintasi khatulistiwa yang mengakibatkan angin bertiup lemah dan bergerak tak menentu.[278][279] Meskipun demikian, tidak semua wilayah Indonesia memiliki pola curah hujan yang sama. Selain daerah musonal, ada pula daerah ekuatorial yang dipengaruhi daerah pertemuan angin antartropis, serta daerah lokal yang polanya berkebalikan dengan pola musonal.[280][281]

Beberapa penelitian memproyeksikan Indonesia terdampak perubahan iklim.[282] Dampak buruk yang ditimbulkan di antaranya kenaikan suhu rata-rata sekitar 1 °C pada pertengahan abad ini akibat emisi yang tidak berkurang,[283][284] peningkatan frekuensi kekeringan dan kekurangan pangan (akibat perubahan curah hujan dan pola musim yang memengaruhi pertanian), serta berbagai penyakit dan kebakaran hutan.[284] Naiknya permukaan air laut juga mengancam sebagian besar penduduk Indonesia yang tinggal di daerah pesisir.[284][285][286] Penduduk prasejahtera mungkin merupakan kelompok yang paling terpengaruh oleh perubahan iklim.[287]

Geologi
Artikel utama: Geologi Indonesia
Artikel utama: Daftar gempa bumi di Indonesia dan Daftar gunung berapi di Indonesia

Gunung-gunung berapi utama di Indonesia, yang berada di antara Cincin Api Pasifik dan Sabuk alpida

Gunung Merapi gunung berapi paling aktif di Indonesia

Kehancuran pada Gempa bumi Yogyakarta 2006
Secara tektonik, sebagian besar wilayah Indonesia sangat tidak stabil karena lokasinya menjadi pertemuan dari beberapa lempeng tektonik, seperti lempeng Indo-Australia, Lempeng Pasifik, dan Lempeng Eurasia. Negara ini terletak di antara Cincin Api Pasifik dan Sabuk alpida sehingga memiliki banyak gunung berapi dan sering mengalami gempa bumi.[288] Busur vulkanik berjajar mulai dari Sumatra, Jawa, Bali dan Nusa Tenggara, dan kemudian ke Kepulauan Banda di Maluku hingga ke timur laut Sulawesi.[289] Dari sekitar 400 gunung berapi, kurang lebih 130 di antaranya masih aktif.[288]

Sebuah letusan supervulkan pada sekitar 77.000 SM yang membentuk Danau Toba dipercaya mengakibatkan musim dingin vulkanik dan penurunan suhu dunia selama bertahun-tahun.[290] Letusan Tambora pada tahun 1815 dan letusan Krakatau pada 1883 juga termasuk letusan gunung terbesar yang tercatat sepanjang sejarah.[291][292]

Gempa bumi terjadi hampir setiap hari di Indonesia dimana sebagian besar tidak dirasakan manusia. Peristiwa gempa bumi besar di Indonesia baru-baru ini adalah Gempa bumi dan tsunami Sulawesi 2018 menewaskan setidaknya 4.300 jiwa. Guncangan mematikan juga terjadi gempa bumi Yogyakarta pada 27 Mei 2006, menewaskan setidaknya 5.700 jiwa, dan menghancurkan ratusan ribu rumah.

Peristiwa Gempa bumi berdorongan besar yang berdampak ke Indonesia dan terjadi belum lama ini adalah gempa bumi dan tsunami Samudra Hindia 2004, dan menyebabkan tsunami besar yang juga berdampak pada negara lain.[293] Indeks resiko dunia menempatkan Indonesia, sebagai negara paling rentan terhadap bencana alam ke-tiga di dunia dengan skor 43 persen.

Lingkungan hidup
Artikel utama: Flora Indonesia, Fauna Indonesia, dan Kawasan perlindungan di Indonesia




Spesies-spesies flora dan fauna yang endemik di Indonesia. Searah jarum jam dari kiri atas: Padma raksasa, orang utan, cenderawasih kuning-besar, dan komodo.
Wilayah Indonesia memiliki keanekaragaman makhluk hidup yang tinggi sehingga dikelompokkan sebagai salah satu dari 17 negara megadiversitas oleh Conservation International.[294][295] Dari sudut pandang wilayah biogeografi, Indonesia termasuk dalam wilayah Malesia. Flora dan faunanya merupakan campuran dari spesies khas Asia dan Australasia. Alfred Russel Wallace, seorang ahli sejarah alam, menghipotesiskan sebuah garis pemisah (yang kemudian disebut garis Wallace) untuk membedakan organisme yang berasal dari Asia (Paparan Sunda) dan Australia (Paparan Sahul). Kawasan biogeografi yang menjadi zona transisi di antara kedua paparan ini disebut Wallacea.[296] Selain itu, garis Weber dan garis Lydekker juga digunakan untuk menetapkan batas biogeografi Indonesia.[297]

Indonesia memiliki sekitar 10% dari seluruh spesies tumbuhan berbunga di Bumi (sebanyak 25.000 spesies, 55% di antaranya endemik di Indonesia). Negara ini juga memiliki sekitar 12% spesies mamalia di Bumi (515 spesies) sehingga menempati peringkat kedua pada keanekaragaman mamalia setelah Brasil. Indonesia menempati peringkat keempat pada keanekaragaman spesies reptil (781 spesies) dan primata (35 spesies), peringkat kelima pada keanekaragaman spesies burung (1.592 spesies), serta peringkat keenam pada keanekaragaman spesies amfibi (270 spesies).[298]


Visibilitas yang rendah di langit Kota Bukittinggi, Sumatera Barat, yang disebabkan oleh kabut asap.
Meskipun demikian, populasi penduduk Indonesia yang besar dan terus meningkat serta industrialisasi yang pesat memunculkan masalah lingkungan hidup yang serius, di antaranya perusakan lahan gambut, penebangan ilegal berskala besar (yang mengakibatkan kabut asap di beberapa bagian Asia Tenggara), eksploitasi sumber daya laut yang berlebihan, polusi udara, pengelolaan sampah, hingga penyediaan air dan sanitasi yang memadai.[299] Isu-isu tersebut berkontribusi pada rendahnya peringkat Indonesia (nomor 116 dari 180 negara) dalam Indeks Kinerja Lingkungan 2020. Laporan tersebut juga menunjukkan bahwa kinerja Indonesia secara umum di bawah rata-rata, baik dalam konteks regional maupun global.[300]

Pada tahun 2018, sekitar 49,7% dari luas daratan Indonesia ditutupi oleh hutan,[301] turun dari angka 87% yang dihitung pada tahun 1950.[302] Sejak dasawarsa 1970-an hingga saat ini, produksi kayu bulat serta berbagai tanaman perkebunan dan pertanian bertanggung jawab atas sebagian besar penebangan hutan di Indonesia.[302] Belakangan ini, penebangan hutan didorong oleh industri kelapa sawit. Meskipun dapat meningkatkan kesejahteraan masyarakat setempat, industri ini dapat merusak ekosistem dan menimbulkan masalah sosial.[303] Situasi ini menjadikan Indonesia sebagai penghasil emisi gas rumah kaca berbasis hutan terbesar di dunia,[304] serta mengancam kelangsungan hidup spesies asli dan endemik. Uni Internasional untuk Konservasi Alam (IUCN) mengidentifikasi sejumlah spesies yang terancam kritis, termasuk jalak bali,[305] orang utan sumatra,[306] dan badak jawa.[307]

Politik dan pemerintahan
Artikel utama: Politik Indonesia dan Pemerintah Indonesia
Sistem pemerintahan

Gedung MPR/DPR dalam Kompleks Parlemen Republik Indonesia.

Gedung Istana Negara, salah satu dari enam Istana Kepresidenan di Indonesia.
Indonesia merupakan negara kesatuan yang menjalankan pemerintahan republik presidensial multipartai yang demokratis. Konstitusi Indonesia adalah Undang-Undang Dasar Negara Republik Indonesia Tahun 1945 (UUD 1945) yang pada era reformasi mengalami empat kali amendemen sehingga membawa perubahan besar pada kekuasaan legislatif, eksekutif, dan yudikatif.[308] Salah satu perubahan utama adalah pendelegasian kekuasaan dan wewenang kepada berbagai entitas regional sambil tetap menjadi negara kesatuan.[309][310]

Kekuasaan eksekutif dipegang oleh presiden yang dibantu oleh wakil presiden dan kabinet. Presiden Indonesia merupakan kepala negara dan kepala pemerintahan, sekaligus panglima tertinggi Tentara Nasional Indonesia. Presiden dan wakil presiden dapat menjabat selama lima tahun dan dapat dipilih kembali hanya untuk satu kali masa jabatan.[311] Joko Widodo dan Ma'ruf Amin adalah pasangan presiden dan wakil presiden yang terpilih untuk masa jabatan 2019–2024. Mereka memimpin Kabinet Indonesia Maju yang terdiri atas 34 menteri dan sejumlah pejabat setingkat menteri.[312]

Lembaga perwakilan tertinggi yaitu Majelis Permusyawaratan Rakyat (MPR), yang berwenang mengubah dan menetapkan konstitusi, serta melantik dan memberhentikan presiden dan/atau wakil presiden.[313] Lembaga ini berbentuk bikameral yang terdiri dari 575 anggota Dewan Perwakilan Rakyat (DPR) yang berasal dari partai politik, ditambah dengan 136 anggota Dewan Perwakilan Daerah (DPD) yang merupakan wakil provinsi dari jalur independen.[314] Anggota DPR dan DPD dipilih melalui pemilihan umum dengan masa jabatan lima tahun. Fungsi yang dijalankan oleh DPR yaitu legislasi (membentuk undang-undang), anggaran (membahas dan menyetujui Anggaran Pendapatan dan Belanja Negara), dan pengawasan (mengawasi kinerja pemerintah),[315][316] sedangkan DPD merupakan lembaga legislatif yang lebih dikhususkan pada pengelolaan daerah.[317][318] Saat ini, MPR diketuai oleh Bambang Soesatyo,[319] DPR diketuai oleh Puan Maharani,[320] sedangkan DPD diketuai oleh La Nyalla Mattalitti.[321]

Kekuasaan kehakiman dijalankan oleh Mahkamah Agung (MA) dan Mahkamah Konstitusi (MK).[322] Sementara itu, Komisi Yudisial mengawasi kinerja para hakim.[323]

Hubungan luar negeri
Artikel utama: Hubungan luar negeri Indonesia

Susilo Bambang Yudhoyono, Presiden Indonesia ke-6, bersama dengan Barack Obama, Presiden Amerika Serikat ke-44, dalam acara penyambutan tamu negara di Istana Merdeka pada 2010.[324]
Indonesia memiliki 132 perwakilan diplomatik di luar negeri, termasuk 95 kedutaan.[325] Negara ini memiliki kebijakan politik luar negeri "bebas dan aktif", yang berarti bahwa Indonesia tidak berpihak pada blok-blok kekuatan dan persekutuan militer di dunia, sekaligus bersikap aktif dalam menjaga ketertiban dunia, sebagaimana dinyatakan dalam Pembukaan UUD 1945.[326]

Berlawanan dengan Sukarno yang anti-Imperialisme, antipati terhadap kekuatan barat, dan bersitegang dengan Malaysia, hubungan luar negeri sejak "Orde baru"-nya Suharto didasarkan pada ekonomi dan kerja sama politik dengan negara-negara barat.[327] Indonesia menjaga hubungan baik dengan tetangga-tetangganya di Asia, dan Indonesia adalah pendiri ASEAN dan East Asia Summit.

Indonesia menjalin hubungan kembali dengan Republik Rakyat Tiongkok pada tahun 1990, padahal sebelumnya melakukan pembekuan hubungan sehubungan dengan gejolak anti-komunis di awal kepemerintahan Suharto. Indonesia menjadi anggota Perserikatan Bangsa-bangsa sejak tahun 1950,[328] dan pendiri Gerakan Non Blok dan Organisasi Kelompok Islam yang sekarang telah menjadi Organisasi Kerja Sama Islam. Indonesia telah menandatangani perjanjian ASEAN Free Trade Area, Cairns Group, dan World Trade Organization, dan pernah menjadi anggota OPEC, meskipun Indonesia menarik diri pada tahun 2008 sehubungan Indonesia bukan lagi pengekspor minyak mentah bersih. Indonesia telah menerima bantuan kemanusiaan dan pembangunan sejak tahun 1966, terutama dari Amerika Serikat, negara-negara Eropa Barat, Australia, dan Jepang.

Pemerintah Indonesia telah bekerja sama dengan dunia internasional sehubungan dengan pengeboman yang dilakukan oleh militan Islam dan Al-Qaeda.[329] Pemboman besar menimbulkan korban 202 orang tewas (termasuk 164 turis mancanegara) di Kuta, Bali pada tahun 2012.[330] Serangan tersebut dan peringatan perjalanan (travel warnings) yang dikeluarkan oleh negara-negara lain, menimbulkan dampak yang berat bagi industri jasa perjalanan/turis dan juga prospek investasi asing.[331] Tetapi beruntung ekonomi Indonesia secara keseluruhan tidak terlalu dipengaruhi oleh hal-hal tersebut di atas, karena Indonesia adalah negara yang ekonomi domestiknya cukup kuat dan dominan.[butuh rujukan]

Militer
Artikel utama: Tentara Nasional Indonesia

Parade para prajurit Tentara Nasional Indonesia Angkatan Darat.
Tentara Nasional Indonesia terdiri dari TNI–AD, TNI-AL (termasuk Marinir) dan TNI-AU.[332] Berkekuatan 400.000 prajurit aktif, memiliki anggaran 4% dari GDP pada tahun 2006, tetapi terdapat kontroversi bahwa ada sumber-sumber dana dari kepentingan-kepentingan komersial dan yayasan-yayasan yang dilindungi oleh militer.[333] Satu hal baik dari reformasi sejalan dengan mundurnya Suharto adalah mundurnya TNI dari parlemen setelah bubarnya Dwi Fungsi ABRI, walaupun pengaruh militer dalam bernegara masih tetap kuat.[334] Gerakan separatis di sebagian daerah Aceh dan Papua telah menimbulkan konflik bersenjata, dan terjadi pelanggaran HAM serta kebrutalan yang dilakukan oleh kedua belah pihak.[335][336] Setelah 30 tahun perseteruan sporadis antara Gerakan Aceh Merdeka dan militer Indonesia, maka persetujuan gencatan senjata terjadi pada tahun 2005.[337] Di Papua, telah terjadi kemajuan yang mencolok, walaupun masih terjadi kekurangan-kekurangan, dengan diterapkannya otonomi, dengan akibat berkurangnya pelanggaran HAM.[338]

Pembagian administratif
Artikel utama: Pembagian administratif Indonesia
Lihat pula: Provinsi di Indonesia, Daftar kabupaten dan kota di Indonesia, dan Daftar kecamatan dan kelurahan di Indonesia
Bagian ini ditransklusi dari Templat:Peta provinsi Indonesia. [ sunting | versi ]
AcehSumatera
UtaraSumatera
BaratRiauKep. RiauKep.
Bangka
BelitungJambiSumatera
SelatanBengkuluLampungBantenDKIJawa
BaratJawa
TengahDIYJawa
TimurBaliNusa
Tenggara
BaratNusa
Tenggara
TimurKalimantan
BaratKalimantan
TengahKalimantan
UtaraKalimantan
TimurKalimantan
SelatanSulawesi
UtaraSulawesi
TengahGorontaloSulawesi
BaratSulawesi
SelatanSulawesi
TenggaraMalukuMaluku
UtaraPapua
BaratPapua
Barat
DayaPapuaPapua
TengahPapua
PegununganPapua
Selatan
Saat ini, Indonesia terdiri atas 38 provinsi,[339] 416 kabupaten dan 98 kota, 7.024 daerah setingkat kecamatan,[340] atau 81.626 daerah setingkat desa/kelurahan.[341]

Di antara provinsi-provinsi tersebut, sembilan di antaranya memiliki status kekhususan dan/atau keistimewaan. Daerah-daerah tersebut ialah Aceh, Daerah Khusus Ibukota Jakarta, Daerah Istimewa Yogyakarta, Papua Barat, Papua Barat Daya, Papua, Papua Tengah, Papua Pegunungan, dan Papua Selatan.

Tiap provinsi memiliki DPRD Provinsi dan gubernur, tiap kabupaten memiliki DPRD Kabupaten dan bupati, sementara tiap kota memiliki DPRD Kota dan wali kota; semuanya dipilih langsung oleh rakyat melalui pemilihan umum. Hal tersebut tidak berlaku pada DKI Jakarta yang terbagi atas kabupaten administrasi atau kota administrasi yang bukanlah daerah otonom, sehingga DPR Kabupaten atau Kota tidak ada di dalam daerah-daerah tersebut, serta bupati dan wali kotanya adalah pegawai negeri yang ditunjuk oleh Gubernur DKI Jakarta.

Indonesia memperbolehkan penamaan lokal/khusus untuk digunakan pada daerah-daerah administratif di bawah tingkat kabupaten/kota, sesuai dengan Undang-Undang tentang Pemerintahan Daerah. Beberapa contoh di antaranya ialah kalurahan, kapanewon, kemantren, gampong, kampung, nagari, pekon, dan distrik.

Berikut ini merupakan provinsi di Indonesia beserta ibu kotanya.

Sumatra
 Aceh — Banda Aceh
 Sumatera Utara (Sumut) — Medan
 Sumatera Barat (Sumbar) — Padang
 Riau — Pekanbaru
 Jambi — Jambi
 Sumatera Selatan (Sumsel) — Palembang
 Bengkulu — Bengkulu
 Lampung — Bandar Lampung
 Kepulauan Bangka Belitung (Babel) — Pangkalpinang
 Kepulauan Riau (Kepri) — Tanjungpinang
Jawa
 Daerah Khusus Ibukota Jakarta (DKI Jakarta)
 Jawa Barat (Jabar) — Bandung
 Jawa Tengah (Jateng) — Semarang
 Daerah Istimewa Yogyakarta (DIY) — Yogyakarta[342]
 Jawa Timur (Jatim) — Surabaya
 Banten — Serang
Nusa Tenggara
 Bali — Denpasar
 Nusa Tenggara Barat (NTB) — Mataram
 Nusa Tenggara Timur (NTT) — Kupang
Kalimantan
 Kalimantan Barat (Kalbar) — Pontianak
 Kalimantan Tengah (Kalteng) — Palangka Raya
 Kalimantan Selatan (Kalsel) — Banjarbaru
 Kalimantan Timur (Kaltim) — Samarinda
 Kalimantan Utara (Kaltara) — Tanjung Selor
Sulawesi
 Sulawesi Utara (Sulut) — Manado
 Sulawesi Tengah (Sulteng) — Palu
 Sulawesi Selatan (Sulsel) — Makassar
 Sulawesi Tenggara (Sultra) — Kendari
 Gorontalo — Gorontalo
 Sulawesi Barat (Sulbar) — Mamuju
Maluku
 Maluku — Ambon
 Maluku Utara (Malut) — Sofifi
Papua
 Papua — Jayapura
 Papua Barat (Pabar) — Manokwari
 Papua Selatan (Pasel) — Merauke
 Papua Tengah (Papteng) — Nabire
 Papua Pegunungan (Papeg) — Wamena
 Papua Barat Daya (PBD) — Sorong
Ibu kota negara
Artikel utama: Ibu kota Indonesia
Indonesia di IndonesiaJakartaJakartaYogyakartaYogyakartaBukittinggiBukittinggiNusantaraNusantara
Ibu kota negara Indonesia sepanjang sejarah
Hingga saat ini, ibu kota negara Republik Indonesia berkedudukan di Daerah Khusus Ibukota Jakarta.[343] Namun sejak tahun 2019, Pemerintah Indonesia melaksanakan proses pemindahan ibu kota Indonesia ke Ibu Kota Nusantara, yang direncanakan akan diresmikan pada tahun 2024.[344]

Semenjak kemerdekaan Indonesia pada tanggal 17 Agustus 1945, ibu kota negara Indonesia secara de facto berkedudukan di Jakarta. Ibu kota negara sempat dipindahkan ke Yogyakarta pada tanggal 4 Januari 1946 ketika pasukan Pemerintahan Sipil Hindia Belanda (NICA) menduduki Jakarta,[345] kemudian ke Bukittinggi pada tanggal 19 Desember 1948 ketika pemerintah pusat lumpuh karena ditawannya Presiden Soekarno dan Wakil Presiden Hatta oleh pasukan militer Belanda dan tampuk pemerintahan dipegang sementara oleh Pemerintahan Darurat Republik Indonesia (PDRI),[346] lalu kembali lagi ke Yogyakarta pada tanggal 6 Juli 1949 setelah kembalinya Soekarno-Hatta dari penawanan. Pada masa Republik Indonesia Serikat (RIS), ibu kota Negara Bagian Republik Indonesia berkedudukan di Yogyakarta sementara ibu kota federal RIS berada di Jakarta. Setelah kembali ke Negara Kesatuan Republik Indonesia pada tanggal 17 Agustus 1950, ibu kota negara kembali berkedudukan di Jakarta. Pada tanggal 28 Agustus 1961, Pemerintah mengeluarkan Peraturan Presiden Nomor 2 Tahun 1961 yang mengukuhkan status Jakarta sebagai ibu kota negara.[347]

Pada tahun 2019, Presiden Joko Widodo melalui Pemerintah Pusat membuat kajian rancangan,[348] melakukan pencanangan,[349] dan menentukan letak wilayah dari ibu kota baru, yaitu sebagian dari wilayah Kabupaten Penajam Paser Utara dan Kutai Kartanegara.[350] Pemerintah bahkan sempat membentuk tim-tim pelaksana pemindahan ibu kota pada bulan Januari 2020,[351][352] yang akan melaksanakan pembangunan pada pertengahan tahun 2020, tetapi harus ditunda akibat pandemi Covid-19.[353] Pada tanggal 18 Januari 2022, Dewan Perwakilan Rakyat mengesahkan Undang-Undang Ibu Kota Negara, yang berisi pembentukan dan garis besar rencana pembangunan ibu kota baru, yang diberi nama Ibu Kota Nusantara, yang kemudian diundangkan pada tanggal 15 Februari 2022.[354] Upacara simbolis penyatuan tanah ketiga puluh empat provinsi di Indonesia saat itu dilakukan oleh Presiden Jokowi bersama para gubernur dan wakil gubernur se-Indonesia pada tanggal 14 Maret 2022 di Titik Nol Ibu Kota Nusantara.[355]

Ekonomi
Artikel utama: Ekonomi Indonesia

Peta yang menunjukkan Produk Domestik Regional Bruto (PDRB) per kapita provinsi-provinsi Indonesia pada tahun 2008 atas harga berlaku. PDRB per kapita provinsi Kalimantan Timur mencapai Rp100 juta manakala PDRB per kapita Maluku, Maluku Utara, dan Nusa Tenggara Timur kurang dari Rp5 juta.
  Lebih dari Rp100 juta
  Rp50–100 juta
  Rp40–50 juta
  Rp30–40 juta
  Rp20–30 juta
  Rp10–20 juta
  Rp5–10 juta
  Kurang dari Rp5 juta
Sistem ekonomi Indonesia awalnya didukung dengan diluncurkannya Oeang Repoeblik Indonesia (ORI) pada tanggal 30 Oktober 1946 yang menjadi mata uang pertama Republik Indonesia, yang selanjutnya berganti menjadi Rupiah.

Pada masa pemerintahan Orde Lama, Indonesia tidak seutuhnya mengadaptasi sistem ekonomi kapitalis, namun juga memadukannya dengan nasionalisme ekonomi. Pemerintah yang belum berpengalaman, masih ikut campur tangan ke dalam beberapa kegiatan produksi yang berpengaruh bagi masyarakat banyak. Hal tersebut, ditambah pula kemelut politik, mengakibatkan terjadinya ketidakstabilan pada ekonomi negara.[356]

Pemerintahan Orde Baru segera menerapkan disiplin ekonomi yang bertujuan menekan angka inflasi, menstabilkan mata uang, penjadwalan ulang utang luar negeri, dan berusaha menarik bantuan dan investasi asing.[356] Pada era tahun 1970-an harga minyak bumi yang meningkat menyebabkan melonjaknya nilai ekspor, dan memicu tingkat pertumbuhan ekonomi rata-rata yang tinggi sebesar 7% antara tahun 1968 sampai 1981.[356] Reformasi ekonomi lebih lanjut menjelang akhir tahun 1980-an, antara lain berupa deregulasi sektor keuangan dan pelemahan nilai rupiah yang terkendali,[356] selanjutnya mengalirkan investasi asing ke Indonesia khususnya pada industri-industri berorientasi ekspor pada antara tahun 1989 sampai 1997[357] Ekonomi Indonesia mengalami kemunduran pada akhir tahun 1990-an akibat krisis ekonomi tahun 1997 yang melanda sebagian besar Asia pada saat itu,[358] yang disertai pula berakhirnya masa Orde Baru dengan pengunduran diri Presiden Soeharto tanggal 21 Mei 1998.

Tampak depan dan belakang dari uang Rp75.000 yang dikeluarkan pada tahun 2020 sebagai peringatan hari kemerdekaan Republik Indonesia yang ke-75.		Tampak depan dan belakang dari uang Rp75.000 yang dikeluarkan pada tahun 2020 sebagai peringatan hari kemerdekaan Republik Indonesia yang ke-75.
Tampak depan dan belakang dari uang Rp75.000 yang dikeluarkan pada tahun 2020 sebagai peringatan hari kemerdekaan Republik Indonesia yang ke-75.
Saat ini ekonomi Indonesia telah cukup stabil. Pertumbuhan PDB Indonesia tahun 2004 dan 2005 melebihi 5% dan diperkirakan akan terus berlanjut.[359] Namun, dampak pertumbuhan itu belum cukup besar dalam memengaruhi tingkat pengangguran, yaitu sebesar 9,75%.[360][361] Perkiraan tahun 2006, sebanyak 17,8% masyarakat hidup di bawah garis kemiskinan, dan terdapat 49,0% masyarakat yang hidup dengan penghasilan kurang dari AS$ 2 per hari.[362]


Bumbu dan rempah-rempah yang umum dijumpai di Indonesia.
Indonesia mempunyai sumber daya alam yang besar di luar Jawa, termasuk minyak mentah, gas alam, timah, tembaga, dan emas. Indonesia pengekspor gas alam terbesar kelima[363] di dunia, meski akhir-akhir ini ia telah mulai menjadi pengimpor bersih minyak mentah. Hasil pertanian yang utama termasuk beras, teh, kopi, rempah-rempah, dan karet.[364] Zulkifli Hasan, Menteri Perdagangan, menyebutkan bahwa Peraturan Presiden №125/2022 berisi tentang cadangan pangan pemerintah yang menjadi prioritas dalam perekonomian negara.[365]

Sektor jasa adalah penyumbang terbesar PDB, yang mencapai 45,3% untuk PDB 2005. Sedangkan sektor industri menyumbang 40,7%, dan sektor pertanian menyumbang 14,0%.[366] Meskipun demikian, sektor pertanian mempekerjakan lebih banyak orang daripada sektor-sektor lainnya, yaitu 44,3% dari 95 juta orang tenaga kerja. Sektor jasa mempekerjakan 36,9%, dan sisanya sektor industri sebesar 18,8%.[367]

Rekan perdagangan terbesar Indonesia adalah Jepang, Amerika Serikat, dan negara-negara jirannya yaitu Malaysia, Singapura dan Australia.

Meski kaya akan sumber daya alam dan manusia, Indonesia masih menghadapi masalah besar dalam bidang kemiskinan yang sebagian besar disebabkan oleh korupsi yang merajalela dalam pemerintahan. Lembaga Transparency International menempatkan Indonesia sebagai peringkat ke-143 dari 180 negara dalam Indeks Persepsi Korupsi, yang dikeluarkannya pada tahun 2007.[368]

Peringkat internasional
Organisasi	Nama Survei	Peringkat
Heritage Foundation/The Wall Street Journal	Indeks Kebebasan Ekonomi	69 dari 180[369]
The Economist	Indeks Kualitas Hidup	71 dari 111[370]
Reporters Without Borders	Indeks Kebebasan Pers	103 dari 168[371]
Transparency International	Indeks Persepsi Korupsi	98 dari 180[372]
United Nations Development Programme	Indeks Pembangunan Manusia	111 dari 189[373]
Forum Ekonomi Dunia	Laporan Daya Saing Global	45 dari 140[374]
Central Connecticut State University	Peringkat Literasi Membaca	60 dari 61[375]
Transportasi di Indonesia
Transportasi di Indonesia mencakup berbagai moda yang menghubungkan pulau-pulau di seluruh kepulauan ini. Sistem transportasi di Indonesia berkembang pesat untuk memenuhi kebutuhan mobilitas penduduk yang terus meningkat. Artikel ini akan membahas secara komprehensif mengenai transportasi darat, laut, dan udara yang ada di Indonesia, mencakup berbagai moda seperti kereta api, bus, kapal feri, dan pesawat udara.

Transportasi Darat
Kereta Api Indonesia
Kereta api adalah salah satu moda transportasi darat utama di Indonesia, yang dioperasikan oleh PT Kereta Api Indonesia (Persero). Sistem perkeretaapian di Indonesia terutama terkonsentrasi di Pulau Jawa dan Sumatra, dengan jaringan yang menghubungkan kota-kota besar serta daerah-daerah di sekitarnya. Kereta api memainkan peran penting dalam pengangkutan penumpang dan barang di sepanjang jalur yang ada.

KRL Commuter Line
KRL Commuter Line adalah sistem kereta rel listrik yang beroperasi di wilayah Jabodetabek (Jakarta, Bogor, Depok, Tangerang, Bekasi) dan di Solo-Jogja. KRL Commuter Line Jabodetabek adalah tulang punggung transportasi massal di kawasan megapolitan ini, mengangkut jutaan penumpang setiap harinya. Sistem ini terus diperluas untuk mencakup lebih banyak rute dan stasiun. Selain itu, KRL Solo-Jogja juga melayani rute commuter di wilayah tersebut, mempermudah akses antara dua kota budaya penting ini.

MRT Jakarta
Mass Rapid Transit (MRT) Jakarta adalah sistem angkutan cepat pertama di Indonesia, yang beroperasi di Jakarta sejak 2019. Tahap pertama rute MRT Jakarta menghubungkan Lebak Bulus di Jakarta Selatan dengan Bundaran HI di pusat kota. Proyek ini bertujuan untuk mengurangi kemacetan di ibu kota dengan menyediakan alternatif transportasi yang cepat, aman, dan nyaman. Rencana pengembangan lebih lanjut akan memperluas jaringan MRT hingga mencakup area yang lebih luas di Jakarta dan sekitarnya.

LRT Jakarta
Light Rail Transit (LRT) Jakarta adalah sistem angkutan ringan yang saat ini beroperasi di sebagian wilayah Jakarta. Rute LRT Jakarta melayani koridor Kelapa Gading-Velodrome, yang merupakan bagian dari upaya pemerintah untuk meningkatkan layanan transportasi publik di ibu kota. LRT ini dirancang untuk mengurangi ketergantungan pada kendaraan pribadi dan mengurangi kemacetan lalu lintas di jalan-jalan utama Jakarta.

LRT Jabodebek
LRT Jabodebek adalah proyek kereta ringan yang dirancang untuk melayani kawasan Jakarta, Bogor, Depok, dan Bekasi (Jabodebek). Proyek ini dimaksudkan untuk menyediakan alternatif transportasi massal yang efisien dan ramah lingkungan bagi masyarakat di wilayah metropolitan Jakarta yang lebih luas. LRT Jabodebek diharapkan dapat terintegrasi dengan moda transportasi lain, seperti KRL dan MRT, untuk membentuk jaringan transportasi yang terintegrasi.

LRT Palembang
LRT Palembang adalah sistem LRT pertama di luar Jakarta, yang diresmikan pada tahun 2018 untuk mendukung Asian Games yang diselenggarakan di kota tersebut. LRT Palembang menghubungkan Bandara Sultan Mahmud Badaruddin II dengan kawasan olahraga Jakabaring, menyediakan akses transportasi yang cepat dan nyaman bagi penumpang.

Bus AKAP
Bus Antar Kota Antar Provinsi (AKAP) adalah moda transportasi darat yang menghubungkan berbagai kota dan provinsi di Indonesia. Bus AKAP sangat penting dalam menyediakan mobilitas bagi masyarakat yang melakukan perjalanan jarak jauh di berbagai pulau, terutama di Jawa, Sumatra, Bali , Kalimantan, dan Sulawesi. Sistem bus AKAP terus berkembang dengan penambahan rute dan modernisasi armada untuk meningkatkan kenyamanan dan keselamatan penumpang. Sistem Bus AKAP Tersebut Dioperasikan Melalui Jaringan Jaringan Terminal Di Kota Kota Besar Seperti Terminal Pulo Gebang Di Jakarta , Terminal Tirtonadi Di Solo , Terminal Bungurasih Di Surabaya Dan Terakhir Terminal Amplas Di Medan

Jalan Tol
Jalan tol merupakan bagian integral dari infrastruktur transportasi di Indonesia. Jalan tol yang menghubungkan berbagai kota besar di Pulau Jawa, Sumatra, dan beberapa pulau lainnya telah mempercepat waktu perjalanan dan meningkatkan efisiensi logistik. Proyek jalan tol trans-Jawa dan trans-Sumatra adalah dua proyek besar yang telah dan sedang dilakukan untuk menghubungkan seluruh bagian pulau tersebut dari ujung ke ujung.

Jalan Nasional
Jalan nasional adalah jaringan jalan utama yang menghubungkan berbagai daerah di Indonesia. Jalan ini menjadi tulang punggung bagi transportasi darat, baik untuk pengangkutan barang maupun penumpang. Jalan nasional memainkan peran penting dalam mendukung perekonomian negara, terutama dalam hal distribusi barang dan jasa antar wilayah.

Kereta Cepat Whoosh Jakarta-Bandung
Kereta Cepat Whoosh adalah proyek kereta cepat pertama di Indonesia yang menghubungkan Jakarta dan Bandung . Dengan kecepatan hingga 350 km/jam, kereta ini mampu mempersingkat waktu perjalanan antara kedua kota menjadi sekitar 40 menit. Proyek ini merupakan bagian dari upaya pemerintah untuk meningkatkan infrastruktur transportasi di Indonesia, dengan rencana perpanjangan jalur hingga Surabaya di masa mendatang.

Transportasi Laut
Kapal Feri dan Kapal Penumpang Lainnya
Indonesia, sebagai negara kepulauan, sangat bergantung pada transportasi laut untuk menghubungkan berbagai pulau. Kapal feri dan kapal penumpang lainnya memainkan peran penting dalam transportasi antar pulau di seluruh nusantara , baik untuk penumpang maupun barang. Selain itu, kapal-kapal ini juga melayani rute internasional yang menghubungkan Indonesia dengan negara-negara tetangga seperti Malaysia dan Singapura, hingga ke negara-negara lain di dunia.

Transportasi Udara
Pesawat
Pesawat terbang adalah moda transportasi yang sangat vital di Indonesia, terutama mengingat luasnya wilayah negara ini yang terdiri dari ribuan pulau. Transportasi udara memungkinkan perjalanan antar pulau yang cepat dan efisien. Sistem penerbangan domestik di Indonesia sangat luas, dengan banyak maskapai penerbangan yang melayani rute-rute di seluruh penjuru tanah air.

Maskapai Garuda Indonesia
Garuda Indonesia adalah maskapai nasional Indonesia dan salah satu yang tertua di Asia. Garuda Indonesia melayani rute domestik dan internasional, dan telah menerima berbagai penghargaan atas layanan dan keselamatannya. Maskapai ini merupakan ikon transportasi udara Indonesia dan memainkan peran penting dalam menghubungkan Indonesia dengan dunia internasional.

Bandara
Indonesia memiliki banyak bandara internasional dan domestik yang tersebar di seluruh kepulauan. Bandara-bandara ini merupakan gerbang utama bagi transportasi udara, melayani jutaan penumpang setiap tahunnya. Beberapa bandara terbesar dan tersibuk di Indonesia termasuk Bandara Internasional Soekarno-Hatta di Jakarta , Bandara Internasional Ngurah Rai di Bali , dan Bandara Internasional Juanda di Surabaya . Infrastruktur bandara terus ditingkatkan untuk menangani peningkatan jumlah penumpang dan volume kargo yang terus berkembang.

Transportasi di Indonesia terus berkembang seiring dengan kebutuhan mobilitas penduduk yang meningkat dan upaya pemerintah untuk meningkatkan infrastruktur transportasi nasional. Berbagai proyek besar seperti pembangunan jalan tol, pengembangan moda transportasi massal seperti MRT, LRT, dan kereta cepat, serta modernisasi pelabuhan dan bandara adalah bagian dari upaya untuk mendukung pertumbuhan ekonomi dan meningkatkan konektivitas antar daerah di Indonesia.

Demografi
Kependudukan
Artikel utama: Demografi Indonesia

Provinsi-provinsi Indonesia menurut kepadatan penduduk di tahun 2015 (per kilometer persegi)
  10.001 ke atas
  1.001 ke 10.000
  101 ke 1.000
  11 ke 100
  1 ke 10
Menurut sensus 2020, jumlah penduduk Indonesia yaitu 270,20 juta jiwa, yang menjadikannya negara berpenduduk terbanyak keempat di dunia,[376] dengan kepadatan penduduk sebanyak 141 jiwa per km2 dan rerata laju pertumbuhan penduduk sebesar 1,25%.[377] Sebanyak 56,1% penduduk (151,59 juta jiwa) tinggal di Pulau Jawa yang merupakan pulau berpenduduk terbanyak di dunia.[378] Pada tahun 1961, sensus pertama setelah Indonesia merdeka mencatat 97 juta penduduk.[379] Populasi diperkirakan mungkin tumbuh menjadi sekitar 295 juta pada tahun 2030 dan 321 juta pada tahun 2050.[380] Indonesia diperkirakan memiliki usia median 31,1 tahun,[381] dan mulai mengalami bonus demografi, yaitu masa ketika jumlah penduduk usia produktif jauh melebihi penduduk usia nonproduktif.[382]

Sebaran penduduk Indonesia tidak merata, dengan tingkat perkembangan yang bervariasi, mulai dari megakota Jakarta hingga suku-suku tak terjamah di Papua.[383] Pada 2017, sekitar 54,7% populasi tinggal di kawasan perkotaan.[384] Sekitar 8 juta orang Indonesia tinggal di luar negeri; sebagian besar dari mereka menetap di Malaysia, Belanda, Arab Saudi, Uni Emirat Arab, Hong Kong, Singapura, Amerika Serikat, dan Australia.[385]

Secara legal, status kewarganegaraan diatur dalam Undang-Undang Nomor 12 Tahun 2006. Warga Negara Indonesia (WNI) diberikan kartu identitas berupa Kartu Tanda Penduduk (KTP) yang mendaftarkan seseorang di suatu wilayah administratif tertentu. Status kewarganegaraan Indonesia dapat diperoleh malalui kelahiran, adopsi, perkawinan, atau pewarganegaraan.[386]

lbs	
Kota-kota besar di Indonesia
 	Kota	Provinsi	Populasi	 	 	Kota	Provinsi	Populasi
1	Jakarta	Daerah Khusus Ibukota Jakarta	11.038.216	Indonesia
Indonesia	7	Makassar	Sulawesi Selatan	1.482.354
2	Surabaya	Jawa Timur	3.018.022	8	Batam	Kepulauan Riau	1.342.038
3	Bandung	Jawa Barat	2.591.763	9	Pekanbaru	Riau	1.167.599
4	Medan	Sumatera Utara	2.546.452	10	Bandar Lampung	Lampung	1.077.664
5	Palembang	Sumatera Selatan	1.801.367	11	Padang	Sumatera Barat	946.982
6	Semarang	Jawa Tengah	1.702.379	12	Malang	Jawa Timur	889.359
Sumber: Data Direktorat Jenderal Kependudukan dan Pencatatan Sipil (per 31 Desember 2024). Catatan: Tidak termasuk kota satelit.
Suku bangsa
Artikel utama: Suku bangsa di Indonesia

Peta suku bangsa di Indonesia
Indonesia merupakan negara yang kaya akan kelompok etnik, dengan sekitar 1.340 suku bangsa.[387] Sebagian besar penduduk Indonesia merupakan keturunan Bangsa Austronesia,[388] dan terdapat juga kelompok-kelompok suku Melanesia, serta kemungkinan Polinesia dan Mikronesia, terutama di Indonesia bagian timur.[389] Kelompok suku menurut bahasa dan asal daerah, misalnya Suku Melayu, Minangkabau, Jawa, Sunda, Batak, Madura, dan lainnya.[390] Menurut sensus 2010, sekitar 40-42% penduduk merupakan suku Jawa yang menghuni hampir seluruh wilayah Indonesia sebagai akibat program transmigrasi.[391] Meskipun demikian, rasa kebangsaan Indonesia dipegang oleh warga negara Indonesia bersama dengan identitas regional yang kuat.[392]

Istilah bumiputra dan pribumi pernah digunakan untuk menyebut kelompok orang yang berbagi warisan sosial budaya yang sama dan dianggap sebagai penduduk asli Indonesia.[393] Pada tahun 1998, Presiden B.J. Habibie menginstruksikan untuk menghentikan penggunaan istilah pribumi dan nonpribumi dalam kehidupan bernegara.[394][395] Sejumlah etnis Asia daratan, seperti etnis Tionghoa, Arab, dan India, sudah lama datang ke Nusantara dan kemudian menetap dan berasimilasi menjadi bagian dari Nusantara. Sensus 2010 mencatat ada sekitar 5 juta WNI yang dikelompokkan sebagai etnis Tionghoa yang tersebar merata di hampir seluruh wilayah di Indonesia (terutama perkotaan) dan 3 juta jiwa dikelompokkan sebagai etnis Arab yang khususnya berada di Pulau Jawa, Sumatera, sebagian Kalimantan, dan sebagian Sulawesi. Sedangkan untuk orang keturunan India populasinya hanya sekitar ratusan ribu saja yang tersebar di beberapa wilayah di Indonesia seperti Medan, Jakarta, Pekanbaru, dan Banda Aceh. Beberapa tempat khususnya di Kota Medan, terdapat wilayah dengan orang etnis/keturunan India yang cukup signifikan yakni di Little India dan Kampung Madhras. [396]

Bahasa
Artikel utama: Daftar bahasa di Indonesia

Gedung Badan Pengembangan dan Pembinaan Bahasa, lembaga yang menjadi pusat perbendaharaan bahasa-bahasa di Indonesia. Lembaga penyusun Kamus Besar Bahasa Indonesia, kamus ekabahasa bahasa Indonesia resmi.
Indonesia memiliki lebih dari 700 bahasa daerah,[397][398] yang secara umum dipertuturkan oleh mayoritas penduduk Indonesia sebagai bahasa ibu dan bahasa sehari-hari.[399] Sebagian besar bahasa daerah tersebut termasuk dalam rumpun bahasa Austronesia, dan di samping itu, ada lebih dari 270 bahasa Papua yang digunakan di Indonesia bagian timur.[400] Menurut jumlah penuturnya, bahasa daerah yang paling banyak digunakan sehari-hari secara berturut-turut adalah Melayu, Jawa, Sunda, Madura, Batak, Minangkabau, Bugis, Betawi, dan Banjar.[401]

Bahasa resmi negara ini adalah bahasa Indonesia, yang merupakan salah satu dari banyak varietas bahasa Melayu.[402] Bahasa Indonesia diajukan sebagai bahasa persatuan sejak masa pergerakan kemerdekaan Indonesia melalui Sumpah Pemuda dan ditetapkan oleh konstitusi pada 1945.[403] Campur tangan negara terhadap bahasa nasional diselenggarakan melalui Badan Pengembangan dan Pembinaan Bahasa di bawah Kementerian Pendidikan, Kebudayaan, Riset, dan Teknologi Republik Indonesia.[404]

Beberapa bahasa asing diajarkan dalam pendidikan formal. Bahasa Inggris sebagai bahasa internasional telah diperkenalkan kepada para pelajar mulai jenjang pendidikan dasar.[405] Bahasa asing lainnya, seperti bahasa Jerman, Prancis, dan Jepang, diajarkan di sejumlah sekolah sebagai pelengkap pada jenjang sekolah menengah atas.[406] Bagi penganut agama Islam yang menjadi kaum mayoritas di Indonesia,[407] bahasa Arab adalah bahasa asing yang memiliki kedudukan khusus karena harus dipraktikkan dalam ibadah harian tertentu, misalnya salat,[408] dan diajarkan di madrasah ibtidaiah dan jenjang selanjutnya.[409] Meskipun demikian, bahasa Arab tidak menjadi bahasa pergaulan umum sejak periode awal keberadaannya di Indonesia.[410]

Agama
Artikel utama: Agama di Indonesia

Masjid Istiqlal di Jakarta, yang merupakan masjid nasional terbesar di Indonesia.
Meskipun menjamin kebebasan beragama dalam konstitusi,[411] pemerintah hanya mengakui enam agama yaitu: Islam, Protestan, Katolik, Hindu, Buddha, dan Konghucu: sementara itu, penganut agama tradisional ataupun agama-agama lainnya hanya mendapatkan pengakuan terbatas sebagai "penghayat kepercayaan".[412][413] . Dengan 231 juta penganut pada tahun 2018, Indonesia adalah negara berpenduduk mayoritas Muslim terbesar kedua di dunia. Sebanyak sekitar hampir 30 juta penduduk Indonesia atau lebih tepatnya 28,6 juta jiwa menganut agama Kristen, di mana 20,2 juta penduduk merupakan penganut aliran Kristen Protestan sedangkan 8,3 juta penganut Kristen Katolik, 4,7 juta penganut Hindu, 2 juta penganut Buddha, 81 ribu penganut Konghucu, dan 108 ribu penganut aliran kepercayaan lainnya (terutama agama tradisional/lokal).[407] Agama Islam dipeluk oleh hampir seluruh warga Indonesia (sekitar 86,70%), Agama Kristen (Protestan & Katolik) kebanyakkan dipeluk oleh beberapa suku, yakni: Batak, Toraja, Dayak, Nias, Minahasa, Ambon, dan lainnya. Kebanyakan pemeluk Hindu adalah Suku Bali dan Orang keturunan India di Indonesia[414] serta kebanyakan pemeluk Buddha dan Konghucu adalah orang Tionghoa-Indonesia.[415]


Pura Besakih di Karangasem, Bali, yang merupakan pura terbesar di Indonesia.
Penduduk asli Indonesia pada awalnya mempraktikkan animisme, paganisme dan dinamisme lokal, yang merupakan kepercayaan umum bangsa Austronesia. Mereka menyembah roh leluhur dan percaya bahwa roh gaib (hyang) mungkin menghuni tempat-tempat tertentu, seperti pohon besar, batu, hutan, gunung, atau tempat keramat.[145] Contoh kepercayaan asli Indonesia di antaranya Sunda Wiwitan, Kaharingan, dan Kejawen. Mereka memberikan dampak yang signifikan pada penerapan agama-agama lain, seperti abangan Jawa, Hindu Bali, dan Kristen Dayak, yang dipraktikkan sebagai bentuk agama yang kurang ortodoks dan sinkretis.[416][417]

Pengaruh agama Hindu mencapai Nusantara pada awal abad pertama Masehi.[418] Kerajaan Salakanagara di Jawa Barat sekitar tahun 130 merupakan kerajaan terkait India Raya pertama yang tercatat dalam sejarah Nusantara.[419] Agama Buddha tiba sekitar abad ke-6,[420] dan sejarahnya di Indonesia berhubungan erat dengan agama Hindu karena kedua agama ini dianut oleh beberapa kerajaan pada periode yang sama. Nusantara mengalami kebangkitan dan kejatuhan kerajaan Hindu dan Buddha yang kuat dan berpengaruh, seperti Majapahit, Sailendra, Sriwijaya, dan Medang. Meski tidak lagi menjadi mayoritas, agama Hindu dan Buddha tetap memiliki pengaruh besar pada budaya Indonesia.[421][422]

Agama Islam diperkenalkan oleh para pedagang Suni dari mazhab Syafi'i serta para pedagang Sufi dari anak benua India dan Arab Selatan pada awal abad ke-8 M.[423][424] Pada sebagian besar perkembangannya, Islam mengalami pencampuran dan saling memengaruhi budaya yang ada sehingga menghasilkan bentuk Islam dengan ciri tersendiri, seperti adanya pesantren.[425][426] Perdagangan dan aktivitas dakwah seperti yang dilakukan oleh Wali Songo dan penjelajah Tiongkok Cheng Ho, serta kampanye militer oleh beberapa kesultanan membantu mempercepat penyebaran Islam.[424][427]


Gereja Katedral Jakarta, yang menjadi salah satu gereja tertua di Indonesia.
Agama Katolik dibawa oleh para pedagang dan misionaris Portugis seperti Yesuit Fransiskus Xaverius, yang mengunjungi dan membaptis beberapa ribu penduduk setempat.[428][429] Penyebarannya menghadapi kesulitan karena kebijakan Perusahaan Hindia Timur Belanda yang melarang agama dan permusuhan oleh Belanda sebagai akibat dari Perang Delapan Puluh Tahun melawan pemerintahan Katolik Spanyol. Protestantisme, sebagian besar, merupakan hasil dari upaya misionaris Calvinis dan Lutheran selama era kolonial Belanda.[430][431][432] Meskipun keduanya merupakan cabang Kekristenan yang paling umum, ada banyak denominasi lain di negara ini.[433]

Jumlah penganut agama Yahudi cukup besar di Nusantara setidaknya sampai tahun 1945, yang kebanyakan merupakan orang Belanda dan orang Yahudi Baghdadi. Sebagian besar di antara mereka meninggalkan Indonesia setelah proklamasi kemerdekaan dan agama Yahudi tidak pernah mendapatkan status resmi. Saat ini hanya sejumlah kecil orang Yahudi di Indonesia, yang kebanyakan berada di Jakarta dan Surabaya.[434]

Pada tingkat nasional dan regional, kepemimpinan dalam politik dan kelompok masyarakat sipil di Indonesia telah memainkan peran penting dalam hubungan antaragama, baik secara positif maupun negatif. Sila pertama Pancasila yang merupakan landasan filosofis Indonesia, yaitu Ketuhanan yang Maha Esa, sering menjadi pengingat toleransi beragama,[435] meskipun kasus-kasus intoleransi juga telah terjadi.[436] Walaupun Indonesia merupakan negara sekuler (bukan negara yang berlandaskan hukum agama), tetapi sebagian besar orang Indonesia menganggap agama sebagai hal yang esensial dan bagian integral dari kehidupan mereka.[437][438]

Pendidikan dan kesehatan
Artikel utama: Pendidikan di Indonesia dan Kesehatan di Indonesia

Gedung Pusat Universitas Gadjah Mada di Yogyakarta.
Sesuai dengan konstitusi yang berlaku,[439] serta Undang-Undang Nomor 20 Tahun 2003 tentang Sistem Pendidikan Nasional, Pemerintah Indonesia baik pusat maupun daerah wajib mengalokasikan anggaran untuk pendidikan sebesar 20% dari APBN dan APBD di luar gaji pendidik dan biaya kedinasan. Semua penduduk wajib mengikuti program wajib belajar sembilan tahun, yang meliputi enam tahun di sekolah dasar dan tiga tahun di sekolah menengah pertama.[440] Pada 2018, tingkat partisipasi penduduk sebesar 93% untuk pendidikan dasar, 79% untuk pendidikan menengah, dan 36% untuk pendidikan tinggi, sementara tingkat melek huruf adalah 96%.[441] Pemerintah menghabiskan sekitar 3,6% dari PDB atau 20,5% dari anggaran negara (2015) untuk pendidikan.[441] Pada tahun 2018, terdapat lebih dari 4.500 perguruan tinggi di Indonesia,[442] dengan universitas terkemuka (seperti Universitas Indonesia, Institut Teknologi Bandung, Universitas Gadjah Mada, dan lainnya) berlokasi di Pulau Jawa.[443]

Anggaran pemerintah untuk sektor kesehatan adalah sekitar 3,3% dari PDB pada tahun 2016.[444] Sebagai bagian dari upaya mencapai cakupan kesehatan semesta, pemerintah meluncurkan Jaminan Kesehatan Nasional (JKN) pada tahun 2014.[445] Meskipun ada peningkatan yang luar biasa dalam beberapa dekade terakhir seperti meningkatnya angka harapan hidup (dari 62,3 tahun pada tahun 1990 menjadi 71,7 tahun pada tahun 2019)[446] dan penurunan kematian anak (dari 84 kematian per 1.000 kelahiran pada tahun 1990 menjadi 25,4 kematian pada tahun 2017),[447] Indonesia terus-menerus menghadapi berbagai tantangan, seperti kesehatan ibu dan anak, kualitas udara yang rendah, kurang gizi, tingginya tingkat merokok, dan penyakit menular.[448]

Indeks Pembangunan Manusia
Artikel utama: Indeks Pembangunan Manusia Indonesia
Menurut UNDP, Indeks Pembangunan Manusia (IPM) Indonesia mencapai angka 0,707[373] pada Laporan Pembangunan Manusia 2019 untuk perkiraan IPM tahun 2018 dan menempati status tinggi, sedangkan menurut Badan Pusat Statistik (BPS), IPM Indonesia tahun 2022 telah mencapai angka 72,91 (0,729)[449][450] dan menempati status tinggi pada tahun 2016.

Perbedaan IPM yang dilaporkan UNDP melalui Human Development Report (HDR) dengan BPS terletak pada besarnya angka IPM dan perincian. Selama ini, memang perbedaan angka IPM sudah dianggap lazim. Namun, sejak sekitar tahun 2011, perbedaan angka IPM UNDP dan BPS meningkat secara signifikan. Dalam perihal perincian, karena UNDP melaporkan dalam tingkat internasional, laporan IPM Indonesia tidak dilaporkan hingga tingkat yang lebih rendah. Sebaliknya, karena BPS hanya melaporkan di tingkat nasional, BPS lebih memerinci, bahkan hingga IPM di tingkat kota/kabupaten dalam laporan beberapa tahun (laporan IPM hingga tingkat kota/kabupaten jarang). Namun, yang selalu dilaporkan di bawah tingkat nasional tentunya adalah laporan IPM di tingkat provinsi/daerah.

Berikut ini adalah daftar provinsi Indonesia menurut IPM tahun 2023 dibandingkan tahun 2022 menurut BPS.[449][451][450]

Peringkat	Provinsi	IPM 2023	Perubahan (%)
Pembangunan Manusia Sangat Tinggi
1 Steady	 Daerah Khusus Ibukota Jakarta	83,55 (0,835)	Kenaikan 0,78
2 Steady	 Daerah Istimewa Yogyakarta	81,09 (0,810)	Kenaikan 0,44
Pembangunan Manusia Tinggi
3 Steady	 Kepulauan Riau	79,08 (0,771)	Kenaikan 0,60
4 Steady	 Kalimantan Timur	78,20 (0,782)	Kenaikan 0,84
5 Steady	 Bali	78,01 (0,780)	Kenaikan 0,61
6 Steady	 Banten	75,77 (0,757)	Kenaikan 0,52
7 Steady	 Sumatera Barat	75,64 (0,756)	Kenaikan 0,48
8 Steady	 Sumatera Utara	75,13 (0,733)	Kenaikan 0,62
9 Steady	 Sulawesi Utara	75,04 (0,750)	Kenaikan 0,52
10 Steady	 Riau	74,95 (0,749)	Kenaikan 0,50
11 Steady	 Aceh	74,70 (0,747)	Kenaikan 0,59
12 Steady	 Kalimantan Selatan	74,66 (0,746)	Kenaikan 0,66
13 Steady	 Jawa Timur	74,65 (0,746)	Kenaikan 0,60
14 Steady	 Sulawesi Selatan	74,60 (0,746)	Kenaikan 0,64
 Indonesia	74,39 (0,743)	Kenaikan 0,62
15 Steady	 Bengkulu	74,30 (0,730)	Kenaikan 0,62
16 Steady	 Jawa Barat	74,25 (0,742)	Kenaikan 0,62
17 Steady	 Kepulauan Bangka Belitung	74,09 (0,740)	Kenaikan 0,59
18 Steady	 Jambi	73,73 (0,737)	Kenaikan 0,62
19 Steady	 Kalimantan Tengah	73,73 (0,737)	Kenaikan 0,56
20 Steady	 Jawa Tengah	73,39 (0,733)	Kenaikan 0,59
21 Steady	 Sumatera Selatan	73,18 (0,718)	Kenaikan 0,70
22 Steady	 Sulawesi Tenggara	72,94 (0,722)	Kenaikan 0,56
23 Steady	 Kalimantan Utara	72,88 (0,728)	Kenaikan 0,67
24 Steady	 Maluku	72,75 (0,727)	Kenaikan 0,71
25 Steady	 Lampung	72,48 (0,724)	Kenaikan 0,69
26 Steady	 Nusa Tenggara Barat	72,37 (0,723)	Kenaikan 0,72
27 Steady	 Sulawesi Tengah	71,66 (0,716)	Kenaikan 0,65
28 Steady	 Gorontalo	71,25 (0,712)	Kenaikan 0,63
29 Steady	 Maluku Utara	70,98 (0,709)	Kenaikan 0,72
30 Steady	 Kalimantan Barat	70,47 (0,704)	Kenaikan 0,76
Pembangunan Manusia Sedang
31 Steady	 Sulawesi Barat	69,80 (0,698)	Kenaikan 0,61
32 Steady	 Nusa Tenggara Timur	68,40 (0,684)	Kenaikan 0,77
33 Steady	 Papua Barat	67,47 (0,674)	Kenaikan 0,75
34 Steady	 Papua	63,01 (0,630)	Kenaikan 0,85
Budaya
Artikel utama: Budaya Indonesia
Pertunjukan

Wayang kulit, salah satu warisan budaya Jawa.
Indonesia memiliki sekitar 300 kelompok etnis, tiap etnis memiliki warisan budaya yang berkembang selama berabad-abad, dipengaruhi oleh kebudayaan India, Arab, Tiongkok, Eropa, dan termasuk kebudayaan sendiri yaitu Melayu. Contohnya tarian Jawa dan Bali tradisional memiliki aspek budaya dan mitologi Hindu, seperti Wayang Kulit yang menampilkan kisah-kisah tentang kejadian mitologis Hindu Ramayana dan Baratayuda. Banyak juga seni tari yang berisikan nilai-nilai Islam. Beberapa di antaranya dapat ditemukan di daerah Sumatra seperti tari Ratéb Meuseukat, Tari Saman, dan tari Seudati dari Aceh.

Seni pantun, gurindam, dan sebagainya dari pelbagai daerah seperti pantun Melayu, dan pantun-pantun lainnya acapkali dipergunakan dalam acara-acara tertentu yaitu perhelatan, pentas seni, dan lain-lain.

Busana
Artikel utama: Daftar busana daerah Indonesia

Seorang gadis Palembang yang tengah mengenakan songket, salah satu busana tradisional Indonesia.
Di bidang busana warisan budaya yang terkenal di seluruh dunia adalah kerajinan Batik. Beberapa daerah yang terkenal akan industri Batik meliputi Yogyakarta, Surakarta, Cirebon, Pandeglang, Garut, Tasikmalaya, Probolinggo, dan juga Pekalongan. Kerajinan Batik ini pun diklaim oleh negara lain dengan industri Batiknya.[452] Busana asli Indonesia dari Sabang sampai Merauke lainnya dapat dikenali dari ciri-cirinya yang dikenakan di setiap daerah antara lain baju Kurung dengan Songketnya dari Sumatera Barat (Minangkabau), kain Ulos dari Sumatera Utara (Batak), busana Kebaya, busana khas Dayak di Kalimantan, baju Bodo dari Sulawesi Selatan, busana Koteka dari Papua dan sebagainya.

Arsitektur
Artikel utama: Arsitektur Indonesia

Kompleks Candi Prambanan yang menonjolkan arsitektur Indonesia zaman dahulu.
Arsitektur Indonesia mencerminkan keanekaragaman budaya, sejarah, dan geografi yang membentuk Indonesia seutuhnya. Kaum penyerang, penjajah, penyebar agama, pedagang, dan saudagar membawa perubahan budaya dengan memberi dampak pada gaya dan teknik bangunan. Tradisionalnya, pengaruh arsitektur asing yang paling kuat adalah dari India. Tetapi, Tiongkok, Arab, dan sejak abad ke-19 pengaruh Eropa menjadi cukup dominan.

Ciri khas arsitektur Indonesia kuno masih dapat dilihat melalui rumah-rumah adat dan/atau istana-istana kerajaan dari tiap-tiap provinsi. Taman Mini Indonesia Indah, salah satu objek wisata di Jakarta yang menjadi miniatur Indonesia, menampilkan keanekaragaman arsitektur Indonesia itu. Beberapa bangunan khas Indonesia misalnya Rumah Gadang, Monumen Nasional, dan Bangunan Fakultas Teknik Sipil dan Perencanaan di Institut Teknologi Bandung.

Olahraga
Artikel utama: Olahraga Indonesia
Sepak bola dan bulu tangkis, dua olahraga paling populer di Indonesia.		Sepak bola dan bulu tangkis, dua olahraga paling populer di Indonesia.
Sepak bola dan bulu tangkis, dua olahraga paling populer di Indonesia.
Olahraga yang paling populer di Indonesia adalah sepak bola dan bulu tangkis.[butuh rujukan] BRI Liga 1 adalah liga klub sepak bola utama di Indonesia.[butuh rujukan] Olahraga tradisional Indonesia termasuk sepak takraw dan karapan sapi. Di wilayah dengan sejarah perang antar suku, kontes pertarungan diadakan, seperti caci di Flores, dan pasola di Sumba. Pencak silat adalah seni bela diri yang unik yang berasal dari wilayah Indonesia. Seni bela diri ini kadang-kadang ditampilkan pada acara-acara pertunjukkan yang biasanya diikuti dengan musik tradisional Indonesia berupa Gamelan dan seni musik tradisional lainnya sesuai dengan daerah asalnya. Olahraga di Indonesia biasanya berorientasi pada pria dan olahraga spektator sering berhubungan dengan judi yang ilegal di Indonesia.[453]

Di ajang kompetisi multi cabang, prestasi atlet-atlet Indonesia tidak terlalu mengesankan. Di Olimpiade, prestasi terbaik Indonesia diraih pada saat Olimpiade 1992, di mana Indonesia menduduki peringkat 24 dengan meraih 2 emas 2 perak dan 1 perunggu, kelima medali tersebut diraih melalui cabang bulu tangkis. Pada era 1960 hingga 2000, Indonesia merajai bulu tangkis. Atlet-atlet putra Indonesia seperti Rudi Hartono, Liem Swie King, Icuk Sugiarto, Alan Budikusuma, Ricky Subagja, dan Rexy Mainaky merajai kejuaraan-kejuaraan dunia. Rudi Hartono yang dianggap sebagai maestro bulu tangkis dunia, menjadi juara All England terbanyak sepanjang sejarah perbulu tangkisan Indonesia. Ia meraih 8 gelar juara, dengan 7 gelar diraihnya secara berturut-turut. Selain bulu tangkis, atlet-atlet tinju Indonesia juga mampu meraih gelar juara dunia, seperti Elyas Pical, Nico Thomas,[454] dan Chris John.[455] dalam ajang sepak bola internasional, Timnas Indonesia (Hindia Belanda) adalah tim Asia pertama yang berpartisipasi di Piala Dunia pada tahun 1938 di Prancis.[456]

Seni musik
Artikel utama: Musik Indonesia
Duration: 25 detik.0:25
Permainan musik angklung.
Seni musik di Indonesia, baik tradisional maupun modern sangat banyak terbentang dari Sabang hingga Merauke. Setiap provinsi di Indonesia memiliki musik tradisional dengan ciri khasnya tersendiri. Musik tradisional termasuk juga Keroncong yang berasal dari keturunan Portugis di daerah Tugu, Jakarta,[457] yang dikenal oleh semua rakyat Indonesia bahkan hingga ke mancanegara. Ada juga musik yang merakyat di Indonesia yang dikenal dengan nama dangdut yaitu musik beraliran Melayu modern yang dipengaruhi oleh musik India sehingga musik dangdut ini sangat berbeda dengan musik tradisional Melayu yang sebenarnya, seperti musik Melayu Deli, Melayu Riau, dan sebagainya.

Alat musik tradisional yang merupakan alat musik khas Indonesia memiliki banyak ragam dari pelbagai daerah di Indonesia, namun banyak pula alat musik tradisional Indonesia yang diklaim oleh negara lain[458] untuk kepentingan penambahan budaya dan seni musiknya sendiri dengan mematenkan hak cipta seni dan warisan budaya Indonesia ke lembaga Internasional UNESCO. Alat musik tradisional Indonesia antara lain meliputi:

Angklung
Bende
Calung
Dermenan
Gamelan
Gandang Tabuik
Gendang Bali
Gendang Karo
Gondang Batak
Gondang (musik Sunda)
Gong Kemada
Gong Lambus
Jidor
Kecapi Suling
Kecapi Batak
Kendang Jawa
Kenong
Kulintang
Rebab
Rebana
Saluang
Sapeh
Saron
Sasando
Serunai
Seurune Kale
Suling Lembang
Suling Batak
Suling Sunda
Talempong
Tanggetong
Tifa, dan sebagainya
Kuliner
Artikel utama: Masakan Indonesia

Nasi goreng, salah satu makanan yang berasal dari Indonesia.
Masakan Indonesia bervariasi bergantung pada wilayahnya.[459] Nasi adalah makanan pokok dan dihidangkan dengan lauk daging dan sayur. Bumbu (terutama cabai), santan, ikan, dan ayam adalah bahan yang penting.[460]

Sepanjang sejarah, Indonesia telah menjadi tempat perdagangan antara dua benua. Ini menyebabkan terbawanya banyak bumbu, bahan makanan dan teknik memasak dari bangsa Melayu sendiri, India, Timur tengah, Tionghoa, dan Eropa. Semua ini bercampur dengan ciri khas makanan Indonesia tradisional, menghasilkan banyak keanekaragaman yang tidak ditemukan di daerah lain. Bahkan bangsa Spanyol dan Portugis, telah mendahului bangsa Belanda dengan membawa banyak produk dari dunia baru ke Indonesia.[butuh rujukan]

Sambal, sate, bakso, soto, dan nasi goreng adalah beberapa contoh makanan yang biasa dimakan masyarakat Indonesia setiap hari.[461] Selain disajikan di warung atau restoran, terdapat pula aneka makanan khas Indonesia yang dijual oleh para pedagang keliling menggunakan gerobak atau pikulan. Pedagang ini menyajikan bubur ayam, mie ayam, mi bakso, mi goreng, nasi goreng, aneka macam soto, siomay, sate, nasi uduk, dan lain-lain.

Rumah makan Padang yang menyajikan nasi Padang, yaitu nasi disajikan bersama aneka lauk-pauk Masakan Padang, mudah ditemui di berbagai kota di Indonesia.[butuh rujukan] Selain itu Warung Tegal yang menyajikan masakan Jawa khas Tegal dengan harga yang terjangkau juga tersebar luas.[butuh rujukan] Nasi rames atau nasi campur yang berisi nasi beserta lauk atau sayur pilihan dijual di warung nasi di tempat-tempat umum, seperti stasiun kereta api, pasar, dan terminal bus. Di Daerah Istimewa Yogyakarta dan sekitarnya dikenal nasi kucing sebagai nasi rames yang berukuran kecil dengan harga murah, nasi kucing sering dijual di atas angkringan, sejenis warung kaki lima. Penganan kecil semisal kue-kue banyak dijual di pasar tradisional. Kue-kue tersebut biasanya berbahan dasar beras, ketan, ubi kayu, ubi jalar, terigu, atau sagu.

Perfilman
Artikel utama: Perfilman Indonesia

Poster Film Loetoeng Kasaroeng.
Film pertama yang diproduksi pertama kalinya di nusantara adalah film bisu tahun 1926 yang berjudul Loetoeng Kasaroeng dan dibuat oleh sutradara Belanda G. Kruger dan L. Heuveldorp pada zaman Hindia Belanda.[butuh rujukan] Film ini dibuat dengan aktor lokal oleh Perusahaan Film Jawa NV di Bandung dan muncul pertama kalinya pada tanggal 31 Desember, 1926 di teater Elite and Majestic, Bandung. Setelah itu, lebih dari 2.200 film diproduksi. Pada masa awal kemerdekaan, sineas-sineas Indonesia belum banyak bermunculan. Di antara sineas yang ada, Usmar Ismail adalah salah satu sutradara paling produktif, dengan film pertamanya Harta Karun (1949).[butuh rujukan] Namun kemudian film pertama yang secara resmi diakui sebagai film pertama Indonesia sebagai negara berkedaulatan adalah film Darah dan Doa (1950) yang disutradarai Usmar Ismail. Dekade 1970 hingga 2000-an, Arizal muncul sebagai sutradara film paling produktif. Tak kurang dari 52 buah film dan 8 judul sinetron dengan 1.196 episode telah dihasilkannya.[butuh rujukan]

Popularitas industri film Indonesia memuncak pada tahun 1980-an dan mendominasi bioskop di Indonesia,[462] meskipun kepopulerannya berkurang pada awal tahun 1990-an. Antara tahun 2000 hingga 2005, jumlah film Indonesia yang dirilis setiap tahun meningkat.[462] Film Laskar Pelangi (2008) yang diangkat dari novel karya Andrea Hirata menjadi film terlaris Indonesia dengan jumlah penonton terbanyak sepanjang sejarah perfilman Indonesia hingga tahun 2016.[463]

Kesusastraan
Sastrawan ternama di Indonesia
Chairil Anwar
Artikel utama: Sastra Indonesia
Bukti tulisan tertua di Indonesia adalah berbagai prasasti berbahasa Sanskerta pada abad ke-5 Masehi.[464] Figur penting dalam sastra modern Indonesia termasuk: pengarang Belanda Multatuli yang mengkritik perlakuan Belanda terhadap Indonesia selama zaman penjajahan Belanda; Muhammad Yamin dan Hamka yang merupakan penulis dan politikus pra-kemerdekaan;[465] dan Pramoedya Ananta Toer, pembuat novel Indonesia yang paling terkenal.[466][467] Selain novel, sastra tulis Indonesia juga berupa puisi, pantun, dan sajak. Chairil Anwar adalah penulis puisi Indonesia yang paling ternama. Banyak orang Indonesia memiliki tradisi lisan yang kuat, yang membantu mendefinisikan dan memelihara identitas budaya mereka.[468]

Kebebasan pers dan media publik
Artikel utama: Kebebasan pers di Indonesia

Logo TVRI pada tahun 1983–1999.
Kebebasan pers di Indonesia meningkat setelah berakhirnya kekuasaan Presiden Soeharto. Jaringan televisi publik TVRI bersaing dengan jaringan televisi swasta nasional dan stasiun daerah; begitu pula dengan jaringan radio publik RRI yang bersaing dengan jaringan radio swasta yang menyiarkan berita dan hiburan.

Internet
Artikel utama: Internet di Indonesia
Pada tahun 2007, dilaporkan bahwa 20 juta penduduk Indonesia menjadi pengguna internet.[469] Kemudian pada tahun 2014, jumlah pengguna internet bertambah pesat menjadi 83,7 juta orang atau terbanyak keenam di dunia.[470] Pada tahun 2019, yaitu tahun sebelum pandemi Covid-19 berlangsung, diperkirakan jumlah pengguna internet di Indonesia adalah 175 juta jiwa. Sementara pada tahun 2022, pengguna internet di Indonesia telah menyentuh angka 210 juta jiwa, yaitu sekitar 77% dari jumlah penduduk Indonesia.[471]
"""

# 1. Bagi konteks menjadi chunk
chunks = chunk_context(context)

# 2. Pre-tokenize semua chunk agar lebih cepat saat tanya
def prepare_inputs_for_chunks(question, chunks):
    inputs = []
    for chunk in chunks:
        input = tokenizer.encode_plus(
            question,
            chunk,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512
        )
        inputs.append((input, chunk))
    return inputs

# 3. Jawab berdasarkan chunk paling relevan
def answer_question(question, chunks):
    best_answer = ""
    max_score = float("-inf")

    prepared_inputs = prepare_inputs_for_chunks(question, chunks)

    for inputs, chunk in prepared_inputs:
        with torch.no_grad():
            outputs = model(**inputs)

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        answer_start = torch.argmax(start_logits)
        answer_end = torch.argmax(end_logits) + 1

        score = start_logits[0, answer_start] + end_logits[0, answer_end - 1]

        if score > max_score:
            max_score = score
            input_ids = inputs["input_ids"][0]
            answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
            best_answer = answer

    return best_answer.strip()

# Loop interaktif
while True:
    question = input("\nTanya: ")
    if question.lower() in ['exit', 'keluar']:
        break
    answer = answer_question(question, chunks)
    print("Jawab:", answer)
