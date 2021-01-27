$( document ).ready(function() {
    var template = $('#abs-btn-template').html();
    $('.abs-btn-container').html(template);
  });
  
  window.onscroll = function() {scrollFunction()};
  
  function scrollFunction() {
    if (document.body.scrollTop > 30 || document.documentElement.scrollTop > 30) {
      document.getElementById("title").style.opacity = "0";
      document.getElementById("title").style.height = "0px";
      document.getElementById("header").style["padding-top"] = "5px";
      document.getElementById("header").style["padding-bottom"] = "0px";
    } else {
      document.getElementById("title").style.opacity = "100";
      document.getElementById("title").style.height = "auto";
      document.getElementById("header").style["padding-top"] = "40px";
      document.getElementById("header").style["padding-bottom"] = "10px";
    }
  }
  
  function absButtonClick(elem){
    var parent = elem.parentNode.parentNode;
    var abstract = parent.parentNode.getElementsByClassName("abs-text")[0];
    var cross = parent.getElementsByClassName("abs-btn-cross")[0];
    if (abstract.style["max-height"] != "1000px") {
      abstract.style["max-height"] = "1000px";
      abstract.style["margin-top"] = "1.5rem";
      abstract.style["margin-bottom"] = "3rem";
      abstract.style.opacity = "1";
      cross.style["transform"] = "rotate(-45deg)";
    } else {
      abstract.style["max-height"] = "0px";
      abstract.style["margin-top"] = "0rem";
      abstract.style["margin-bottom"] = "0rem";
      abstract.style.opacity = "0";
      cross.style["transform"] = "rotate(0deg)";
    }
  }
  