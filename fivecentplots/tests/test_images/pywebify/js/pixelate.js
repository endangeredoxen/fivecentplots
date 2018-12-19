/*******************************************************************************/
/* Pixelate checkbox toggler                                                   */
/*   Designed for the PyWebify project                                         */
/*   https://github.com/endangeredoxen/pywebify                                */
/*******************************************************************************/


function img_pix_toggle() {

    /* Add div */
    $('#collapse').prepend('<div id="pixelCheckboxDiv"></div>');

    /* Add checkbox */
    $('#pixelCheckboxDiv').append($('<label>',
                          { id : "pixelCheckboxLabel"}));


    $('#pixelCheckboxLabel').append($('<input>',
                          { id : "pixelCheckbox", type:"checkbox",
                           name: "pixelCheckbox"}));
    $('#pixelCheckbox').prop('checked', true);

    /* Add text */
    $('#pixelCheckboxLabel').append('<span id="pixelCheckboxSpan"></span>');
    $('#pixelCheckboxLabel').append('<span id="pixelCheckboxText">Pixel Smoothing</span>');

    /* Check event */
    $('#pixelCheckbox').change(function(){
    if($(this).is(":checked")){
        $("img").css('image-rendering','auto');
    }
    else if ($.browser.mozilla) {
        $("img").css('image-rendering','-moz-crisp-edges');
    }
    else{
        $("img").css('image-rendering','pixelated');

    }
    });

}

$(document).ready( function() {
    img_pix_toggle()
});