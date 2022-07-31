/*******************************************************************************/
/* Div size toggler                                                            */
/*   Designed for the PyWebify project                                         */
/*   https://github.com/endangeredoxen/pywebify                                */
/*******************************************************************************/

function div_toggle() {
    var origWidth = $('#sidebar').width();
    $('#toggle').html('<<');
    $('#toggle').toggle(function(){
    $('#sidebar').css({'overflow': 'hidden'});
    $('#sidebar li').css({'visibility': 'hidden'});
    $('#sidebar').animate({width:0});
    $('#viewer').animate({left:0});
    $('#toggle').html('>>');
    },function(){
    $('#sidebar').css({'overflow': 'auto'});
    $('#sidebar li').css({'visibility': 'visible'});
    $('#sidebar').animate({width:origWidth});
    $('#toggle').html('<<');
    });
    
    // position the toggle at the bottom and account for a reduced viewer div height
//    var h1 = $('#viewer').height();
//    $('#viewer').height('100%');
//    var h2 = $('#viewer').height();
//    var h3 = h2-h1;
//
//    $('#toggle').css({'bottom':h3+'px'});
//    $('#viewer').height(h1);
};

$(document).ready( function() {
    div_toggle()
});
