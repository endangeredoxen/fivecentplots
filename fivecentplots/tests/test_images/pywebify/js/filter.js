/*******************************************************************************/
/* Filters the report file list                                                */
/*   Modified from http://jsfiddle.net/GoranMottram/4CJMe/4/                   */
/*   Designed for the PyWebify project                                         */
/*   https://github.com/endangeredoxen/pywebify                                */
/*******************************************************************************/
(function ($) {

    function searchList(elm, mylist) { // elm is any element, mylist is an unordered list
        // create and add the filter form to the elm
        var form = $("<form>").attr({"class":"filterform","action":"#"}),
            input = $("<input>").attr({"class":"filterinput","type":"text","value":"Search..."});
        $(form).append(input).appendTo(elm);
        $(input).css({marginTop:$(elm).height()/2-$(input).outerHeight()/2}); //center the input field vertically
        
        $(input).keyup(function(){
            var searchTerms = $(this).val();
            $('#expandList').click();
            console.log("Input change")
            $(mylist + ' li').each(function() {
              var hasMatch = searchTerms.length == 0 || $(this).text().toLowerCase().indexOf(searchTerms.toLowerCase()) > 0;

              $(this).toggle(hasMatch);
            });
        });
        $(input).focus(function() {
            $(input).val("");
        });
    }

    //ondomready
    $(function () {
        searchList($("#navbar"), "#collapse");
    });
}(jQuery));