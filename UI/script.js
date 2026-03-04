Dropzone.autoDiscover = false;

function init() {
    let dz = new Dropzone("#dropzone", {
        url: "/", // Dropzone'un kendi kendine yüklemesini engellemek için zararsız bir URL
        maxFiles: 1,
        addRemoveLinks: true,
        dictDefaultMessage: "Some Message",
        autoProcessQueue: false // ÖNEMLİ: Kendi kendine yüklemeyi durdurur
    });
    
    dz.on("addedfile", function() {
        if (dz.files[1]!=null) {
            dz.removeFile(dz.files[0]);        
        }
    });

    // Butona tıklandığında Dropzone değil, bizim belirlediğimiz jQuery Ajax çalışacak
    $("#submitBtn").on('click', function (e) {
        e.preventDefault(); // Sayfanın veya formun yenilenmesini engeller

        if (dz.files.length === 0) {
            alert("Lütfen önce bir resim yükleyin.");
            return;
        }

        let file = dz.files[0];
        let imageData = file.dataURL; // Base64 formatındaki resim verisi
        
        // Flask sunucumuza giden yol
        let url = "/classify_image";

        // Ajax ile veriyi gönderiyoruz
        $.post(url, {
            image_data: imageData
        }, function(data, status) {
            console.log("Sunucudan gelen veri:", data);
            
            // Eğer yüz bulunamazsa veya boş veri dönerse hata mesajını göster
            if (!data || data.length === 0) {
                $("#resultHolder").hide();
                $("#divClassTable").hide();                
                $("#error").show();
                return;
            }
            
            let match = null;
            let bestScore = -1;
            
            // En yüksek olasılığa sahip oyuncuyu bul
            for (let i = 0; i < data.length; ++i) {
                let maxScoreForThisClass = Math.max(...data[i].class_probability);
                if(maxScoreForThisClass > bestScore) {
                    match = data[i];
                    bestScore = maxScoreForThisClass;
                }
            }
            
            // Sonucu ekrana yazdır
            if (match) {
                $("#error").hide();
                $("#resultHolder").show();
                $("#divClassTable").show();
                
                // Oyuncu resmini ve ismini göster
                let playerCardHtml = $(`[data-player="${match.class}"]`).html();
                $("#resultHolder").html(playerCardHtml);
                
                // Tablodaki olasılık skorlarını doldur
                let classDictionary = match.class_dictionary;
                for(let personName in classDictionary) {
                    let index = classDictionary[personName];
                    let probabilityScore = match.class_probability[index];
                    let elementName = "#score_" + personName;
                    $(elementName).html(probabilityScore);
                }
            }
        }).fail(function(jqXHR, textStatus, errorThrown) {
            // Eğer Python tarafında (OpenCV vb.) bir 500 hatası olursa konsola yazdır
            console.error("Sunucu Hatası:", textStatus, errorThrown);
            alert("Resim işlenirken sunucuda bir hata oluştu. Lütfen terminal (komut satırı) ekranındaki hatayı kontrol edin.");
        });
    });
}

$(document).ready(function() {
    console.log( "Sayfa Hazır!" );
    $("#error").hide();
    $("#resultHolder").hide();
    $("#divClassTable").hide();

    init();
});