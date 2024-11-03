$(document).ready(function () {
  // Init
  $(".loader").hide();
  $("#btn-predict").hide();
  $("#result").hide();

  // Upload Preview
  function readURL(input) {
    if (input.files && input.files[0]) {
      var reader = new FileReader();
      reader.onload = function (e) {
        $("#imagePreview").css("background-image", "url(" + e.target.result + ")");
        $("#imagePreview").hide();
        $("#imagePreview").fadeIn(650);
      };
      reader.readAsDataURL(input.files[0]);
    }
  }
  $("#imageUpload").change(function () {
    $("#btn-predict").show();
    $("#result").text("");
    $("#result").hide();
    readURL(this);
  });

  // Predict
  $("#btn-predict").click(function () {
    var form_data = new FormData($("#upload-file")[0]);

    // Show loading animation
    $(this).hide();
    $(".loader").show();

    // Make prediction by calling api /predict
    $.ajax({
      type: "POST",
      url: "/predict",
      data: form_data,
      contentType: false,
      cache: false,
      processData: false,
      async: true,
      success: function (data) {
        // Get and display the result
        $(".loader").hide();
        $("#result").fadeIn(600);
        let result = `
            <div class='w-100'>
                <div class='info-item'>
                    <span class='info-key'>- Tên biển:</span>
                    <span class='info-value'>${data.name}</span>
                </div>
                <div class='info-item'>
                    <span class='info-key'>- Loại biển:</span>
                    <span class='info-value'>${data.type}</span>
                </div>
                <div class='info-item'>
                    <span class='info-key'>- Mô tả:</span>
                    <span class='info-value'>${data.description}</span>
                </div>
            </div>
        `;
        $("#result").html(result);
        console.log("Success!");
      },
    });
  });
});
