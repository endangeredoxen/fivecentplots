/*******************************************************************************/
/* Makes the file list ul dynamically expandable/collapsible                   */
/*   Modified from http://jasalguero.com/ledld/development/web/expandable-list */
/*   Designed for the PyWebify project                                         */
/*   https://github.com/endangeredoxen/pywebify                                */
/*******************************************************************************/


function prepareList() {
    $('#collapse').find('li:has(ul)')
    .click( function(event) {
        if (this == event.target) {
            $(this).toggleClass('expanded');
            $(this).children('ul').toggle();
        }
        return false;
    })
    .addClass('collapsed')
    .children('ul').hide();


    //Create the button funtionality
    $('#expandList')
    .unbind('click')
    .click( function() {
        $('.collapsed').addClass('expanded');
        $('.collapsed').children().show();
    })
    $('#collapseList')
    .unbind('click')
    .click( function() {
        location.reload();
    })

};


/**************************************************************/
/* Functions to execute on loading the document               */
/**************************************************************/
$(document).ready( function() {
//    document.getElementById('sidebar').innerHTML = '<div class="listControl"><a id="expandList">Expand All</a><a id="collapseList">Collapse All</a></div>';
    $('#sidebar').prepend('<div class="listControl"><a id="expandList">Expand All</a><a id="collapseList">Collapse All</a></div>');
    prepareList()
});