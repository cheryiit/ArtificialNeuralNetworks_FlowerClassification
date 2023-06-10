## II. FLOWER CLASSIFICATION USING A SIMPLE NEURAL NETWORK

```
YAPAY SİNİR AĞI ile ÇİÇEK SINIFLANDIRMA
```
Doğada gördüğümüz bir çiçeğin, Zambak (veya süsen) bitkisi olduğu biliniyor. Üç farklı
türünden hangisine ait olduğunu bulduran bir algoritmanın yazılması istenmektedir. Elimizde
her bir çiçek türünden 150 örnek üzerinden ölçülerek alınan veriler bulunmaktadır. Her bir
örnek için 4’er adet özellik (çanak yaprak uzunluğu, çanak yaprak genişliği, taç yaprak
uzunluğu, taç yaprak genişliği) ve hangi sınıfta (tür) olduğu bilgisi hazır olarak verilmektedir.
Tablo 3 ’te 6 tanesine yer verilmiştir. Veriseti: https://archive.ics.uci.edu/ml/machine-learning-
databases/iris/iris.data bağlantısında yer almakta olup projede kullanılacaktır.

Tablo 3 : Çiçek veri setinden alınmış 6 adet bitki örneğine ilişkin bilgiler

**Aşağıdaki gibi bir yapay sinir ağı üreterek (sınıfı oluşturarak), verilen çiçek sınıflandırma
probleminin çözümünde kullanınız.** Java veya C# dillerinden herhangi birisini tercih
edebilirsiniz. Hazır Makine öğrenmesi ve Veri Madenciliği kütüphanesinden yararlanmayınız.

ÇU : Çanak yaprak uzunluğu, ÇG: Çanak yaprak genişliği,
TU : Taç yaprak uzunluğu, TG : Taç yaprak genişliği,
N1 : Çıktı nöronu 1 N2 : Çıktı nöronu 2 N3 : Çıktı nöronu 3 olmak üzere,

Makine Öğrenmesi yöntemi olan ve derin öğrenme alanının temelini de oluşturan Yapay Sinir
Ağları (Artificial Neural Networks - ANN) konusundaki en temel yapılar Yapay Sinir
Hücreleridir (Artificial Neuron). ANN’ler sınıflandırma, kümeleme ve tahminleme gibi birçok
problemin çözümünde kullanılırlar.

Yapay sinir hücresinin yapısı ve örnek bir hesaplama işlemi Şekil 1 ’de gösterilmektedir.
Şekildeki nöronun 4 adet girdisi (x) ve 1 adet çıktısı (y) bulunmaktadır.

```
ÇU
```
```
ÇG
```
```
TU
```
```
TG
```
```
Girdi Katmanı
```
```
N
```
```
N
```
```
N
```
```
Çıktı Katmanı
```
```
I_Setosa
```
```
I_Versicolor
```
```
I_Virginica
```
```
Ağırlıklar
```

```
Şekil 1 : Sinir Hücresi (Nöron) Modeli ve İşleyişi
```
```
Toplama İşlevi, girdilerle ağırlıkların çarpımları toplamının alınması şeklinde gerçekleştirilir:
```
```
Gözetimli Öğrenmede (Supervised Learning), girdilerle beraber, olması gereken çıktı değerleri
(target) verilir / sistem tarafından sağlanır. Ağ eğitilirken yukarıda bağlantısı verilen iris.data
veri setinin kullanılması gerekmektedir. Eğitim setinde her biri
5.1,3.5,1.4,0.2,Iris-setosa
formatında 150 adet veri (bitki örneği) bulunmaktadır. Satırdaki ilk 4 değer, ilgili çiçek
örneğinin öznitelikleri olup ağa girdi olarak verilecek, sondaki değer ise ilgili zambak bitkisinin
türü olup çıktıda yer alacak, target yani beklenen değer olarak kullanılacaktır.
```
```
a) Bir Neuron (Sinir Hücresi) sınıfı oluşturunuz. Girdiler ve ağırlıkları tutmak için
uygun veri yapılarını tercih ediniz. Tüm ağırlıkları en başta [ 0 , 1 ] arasında rastgele
(random) pozitif değerlerden oluşturunuz. Hesaplamaları ve gerekli işlemleri yapan
metodu veya metotları yazınız (nöron çıktısını hesaplayan).
b) Neural Network (Yapay Sinir Ağı) Sınıfı ve 3 nöron içerecek nesneyi oluşturunuz.
Beklenen değerler ile nöron çıktılarına bakarak eğitimi yapan metodu yazınız :
Eğitimden önce tüm girdi verilerini 10’a bölerek [0, 1] aralığına ölçeklendiriniz
(Dileyenler bölme işlemini yapmak yerine gerçek normalizasyon işlemi yapabilirler).
Eğitim, basitleştirilmiş bir öğrenme kuralına göre yapılacaktır: Çıktısı I Setosa olan
verilerde N1’in, Çıktısı I Versicolor olan verilerde N2’nin, Çıktısı I Virginica olan
verilerde N3’ün beklenen değeri 1; Diğer çıktıların değeri 0 alınacaktır. İlgili veri için
ağın çıktı değerleri (N1, N2 ve N3) hesaplanacak, beklenen çıktı değeri ile ağın ürettiği
çıktılardan en büyük değere sahip nöron aynı ise işlem yapılmayacak; farklı ise:
A ğın ürettiği çıktıların en büyüğüne bağlı ağırlıkların (w) değerleri, λ öğrenme
katsayısı ve x ise ilgili ağırlığa bağlı girdinin değeri olmak üzere,
```
- w = w **–** ( **λ** * x) **formülü ile azaltılacak
ve beklenen çıktıya bağlı ağırlıkların** (w) **değerleri ise**
- w = w + ( **λ** * x) **formülü ile artırılacaktır.**
c) **Ağı λ (öğrenme katsayısı) = 0.01 olacak şekilde** 5 0 epok boyunca **eğitiniz**. Bir epok,
tüm eğitim verilerinin (burada 150 adet) sisteme bir kere sıra ile verilerek ağırlıkların
değiştirilmesi işlemidir. İşlem bittikten sonra girdi verilerini sadece sonuç elde edecek
şekilde (ağırlıkları değiştirmeden) ağa verip çıktı değerlerini hesaplayınız. Bu aşamada
ağın ürettiği çıktılardan en büyük olanı ile beklenen değeri 1 olan nöron aynı ise doğru
bilinenlerin sayısını bir artırınız. **Doğruluk** değerini (doğru olarak sınıflandırılan **örnek
(veri) sayısı / toplam örnek sayısıdı** r) hesaplayıp yazdırınız. Elinizdeki 150 verinin
120 tanesi doğru olarak sınıflandırıldıysa doğruluk değeri (accuracy) = 120 / 150 = % 80.

```
y = v = 0.
```
```
Üretilen Çıktı
```
10 p

10 p

10 p


```
d) Bu işlemi 20 ve 10 0 epok için de baştan tekrarlayınız. λ = 0.0 05 ve 0. 025 için ( 20 , 50
ve 100 epok) için deneyleri tekrarlayınız. Epok sayıları ( 20 , 5 0, 10 0) satırlarda, λ
değerleri (0.0 05 , 0.0 1 ve 0. 025 ) sütunlarda olmak üzere doğruluk değerlerini 3x3’lük
bir tabloya kaydederek raporda sununuz. c ve d maddelerindeki deneyleri 2 kere daha
tekrarlayarak sonuçları 2 yeni tablo daha oluşturunuz. Başarı değerlerinin değişip
değişmediğini gözlemleyiniz. Nedenini düşününüz.
