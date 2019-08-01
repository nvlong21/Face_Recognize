function openAct(evt, actName) {
  var i, tabcontent, tablinks;
  tabcontent = document.getElementsByClassName("tabcontent");
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }
  tablinks = document.getElementsByClassName("tablinks");
  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(" active", "");
  }
  document.getElementById(actName).style.display = "block";
  evt.currentTarget.className += " active";
}

// Get the element with id="defaultOpen" and click on it
defaultOpen = document.getElementById("defaultOpen");
if (defaultOpen){
  defaultOpen.click();
}
function d_load() {
  document.getElementById("imgOpen").click();
}

function b_nav() {
  window.history.back();
}

function ChangePhoto(name, img) {
  img = typeof img !== 'undefined' ? img : "{{ result['original'] }}";
  target = document.getElementById("label");
  if (target){
    target.innerHTML = name;
    target = document.getElementById("photo");
    target.src = img;
  }
}

function WaitDisplay(upName) {
  target = document.getElementById("result");
  if (target){
    target.style.display = "none";
  }
  target = document.getElementById("loading");
  if (target){
    target.style.display = "";
  }
  setTimeout(function() {
    document.getElementById(upName).submit();
  }, 100);
}
function compare() {
      var img1 = document.getElementById("img_1").src
      var img2 = document.getElementById("img_2").src
      $.ajax({
        type: 'POST',
        url: "/compare_two_img",
        dataType: 'json',
        contentType: 'application/json; charset=utf-8',
        data: JSON.stringify({img_1: img1, img_2: img2}),
        // dataType: "text",
        success: function(data){
                      console.log(data["results"])
                      var html_temp = "<div class=\"row\">\
                            <h2>" + data["results"] + "%</h2>\
                        </div>";
                      var result = document.getElementById("result_compare");
                      result.innerHTML = html_temp;

                    }
      });
}
