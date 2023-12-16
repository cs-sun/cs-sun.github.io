$( document ).ready(function() {
  var template = $('#abs-btn-template').html();
  $('.abs-btn-container').html(template);
});

// window.onscroll = function() {scrollFunction()};

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

// $(window).scroll(function(){
//     $("#navbar").css("top", Math.max(55, 105 - $(this).scrollTop()));
// });

function paperListToggle() {
  var elem = $('#paper-view-toggle');
  $('.hide-when-view-less').not('.year-tag').slideToggle();
  $('.year-tag').fadeToggle();
  if (elem.hasClass("view-all")) {
    elem.removeClass("view-all");
    elem.addClass("view-less");
    elem.text("[view selected]");
  } else {
    elem.removeClass("view-less");
    elem.addClass("view-all");
    elem.text("[view full list]");
  }
}

function paperListShowAll() {
  var elem = $('#paper-view-toggle');
  $('.hide-when-view-less').not('.year-tag').slideDown();
  $('.year-tag').fadeIn();
  elem.removeClass("view-all");
  elem.addClass("view-less");
  elem.hide();
}

function paperListShowLess() {
  var elem = $('#paper-view-toggle');
  $('.hide-when-view-less').not('.year-tag').slideUp();
  $('.year-tag').fadeOut();
  elem.removeClass("view-less");
  elem.addClass("view-all");
  elem.text("[view full list]");
}


$(document).ready(function() {
  $('#paper-view-toggle').click( paperListToggle );

  // $('.section').hide();
  $('.nav-first-level').siblings().hide();
  $('.page-about').addClass('current-page');
  $('.page-all-pages').addClass('current-page');
  $('.current-page').fadeIn();


  $('.view-all').click(function(){
    if ($('#navbar').is(":visible")) {
      $('#navbar a[href="#papers"]').trigger('click');
    } else if ($('#navbar-h').is(":visible")) {
      $('#navbar-h a[href="#papers"]').trigger('click');
    }
  });


  $('.nav-first-level').click(function(){
    if($(this).hasClass('active')){
      return;
    }

    var clickedPagelink = $(this).attr('pagelink');
    var clickedPage = $(clickedPagelink);
    var currentPage = $('.current-page');

    $('.nav-first-level').removeClass('active');
    if ($('#paper-view-toggle').hasClass('view-less')) {
      paperListToggle();
    }
    $('.nav-first-level').siblings().slideUp();

    $(this).addClass('active');
    $(this).siblings().not('.hide-at-first').slideDown();

    currentPage.fadeOut(function(){
      scroll(0,0);
      currentPage.trigger('onHide');
      if (clickedPagelink == '.page-papers') {
        paperListShowAll();
      } else if (clickedPagelink == '.page-about') {
        paperListShowLess();
      }
      clickedPage.fadeIn();
      $('.page-all-pages').fadeIn();
    });
    currentPage.not('.page-all-pages').removeClass('current-page');
    clickedPage.addClass('current-page');
  });


  $('.nav-first-level-h').click(function() {
    if($(this).hasClass('active')){
      return;
    }

    var clickedPagelink = $(this).attr('pagelink');
    var clickedPage = $(clickedPagelink);
    $('#navbar a[pagelink="' + clickedPagelink + '"]').trigger('click');

    $('.nav-first-level-h').removeClass('active');
    $(this).addClass('active');
  });


});





//highlight nav
$(window).scroll(function () {
  
   if($(document).scrollTop() > 20){
    $('#header').css('box-shadow', '0 10px 6px -6px #0000004d');
    $('#header').css('padding-top', '5px');
    $('#header').css('padding-bottom', '0px');
    if(window.matchMedia('(min-width: 900px)').matches) {
      $('#title').css('font-size', '140%');
      // $('#title-container').css('padding-bottom', '5px');
    } else {
      $('#title-container').css('opacity', '0');
      // $('#title-container').css('line-height', '0');
    }
   } else {
    $('#header').css('box-shadow', '0 0px #ffffff');
    $('#header').css('padding-top', '20px');
    $('#header').css('padding-bottom', '10px');
    if(window.matchMedia('(min-width: 900px)').matches) {
      $('#title').css('font-size', '160%');
    } else {
      $('#title-container').css('opacity', '1');
      // $('#title-container').css('line-height', '1.7');
    }
   }

  //  var position = window.pageYOffset;
   $('.anchor').each(function () {
     // $('#navbar a').removeClass('active');
     // if (!navLinks.prop('classList').length) navLinks.removeAttr('class');

     var target = $(this).offset().top;
     var id = $(this).attr('id');


     if (id == 'about') {
       $('#navbar .nav-second-level').removeClass('active');
     } else if (position + 100 >= target && $(this).is(":visible")) {
       // if ($(this).hasClass('end-first-level')) {
       //   $('#navbar .nav-first-level').removeClass('active');
       // } else
       if ($(this).hasClass('end-second-level')) {
         $('#navbar .nav-second-level').removeClass('active');
       } else {
           var thisLink = $('#navbar a[href="#' + id + '"]');

           // if (position >= target) {
           // if (thisLink.hasClass('nav-first-level')) {
           //   $('#navbar .nav-first-level').removeClass('active');
           // } else
           if (thisLink.hasClass('nav-second-level')) {
             $('#navbar .nav-second-level').removeClass('active');
           }

           thisLink.addClass('active');
       }
     }
   });
});
